import argparse
import re
import ast
import networkx as nx
from huggingface_hub import login
import sys
import os
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTConfig,SFTTrainer
import subprocess as sp
from time import perf_counter
import nvidia_smi

# sys.path.append(os.path.abspath('/home/ubuntu/LLMCODE/LLM4DistReconfig/Dataset-Notebooks/utils'))
from dataset_utils import *
from model_utils import *
from generation_utils import *
from monitoring_utils import *

'''
当前代码只能在一个显卡上运行，不能自动在两张卡上运行
https://huggingface.co/docs/trl/v0.17.0/sft_trainer#multi-gpu-training 里面说要加python -m torch.distributed.launch，但实际上并launch不起来
单个显卡显存不够用的
'''

class CustomTrainerllama31(SFTTrainer):

    def __init__(self, train_df=None, cycles_loss_scaling_factor=None,custom_loss_config=None, * args, **kwargs):
        """
        自定义训练器，支持配电网重构任务的特定损失计算

        Args:
            cycles_loss_scaling_factor: cycles损失权重
            train_df: DataFrame,成员变量，存储所有训练样本，用于比较模型输出和最优输出的数学结果
            *args: 父类参数
            **kwargs: 父类关键字参数
        """
        super().__init__(*args, **kwargs)
        self.train_df = train_df  # 存储训练样本的DataFrame

        # 设置默认损失权重 [cycles_loss, unsupply_loss, invalid_loss]
        if cycles_loss_scaling_factor is None:
            self.loss_scaling_factor = [1.0, 0.01, 1.0]  # 默认权重
        else:
            self.loss_scaling_factor = [cycles_loss_scaling_factor, 0.01, 1.0]

        # 设置默认自定义损失函数计算方式
        if custom_loss_config is None:
            self.custom_loss_config = "IEL CYL SUL"
        else:
            self.custom_loss_config = custom_loss_config

    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取模型基础输出和损失
        outputs = model(**inputs)
        loss = outputs.loss

        # 解码输入输出文本
        input_text = self.tokenizer.batch_decode(
            inputs["input_ids"], skip_special_tokens=True
        )[0]
        output_text = self.tokenizer.batch_decode(
            outputs.logits.argmax(dim=-1), skip_special_tokens=True
        )[0]

        # 初始化自定义损失项
        cycles_loss = torch.tensor(0.0)
        unsupply_loss = torch.tensor(0.0)
        invalid_loss = torch.tensor(0.0)

        # 检查输出是否成功
        success_match = re.search(
            r'"success"\s*:\s*(true|false)', output_text, re.IGNORECASE
        )

        try:
            # 如果output解析失败或success=false，使用惩罚值
            if not success_match or success_match.group(1).lower() == "false":
                lines, vsources, switches = parse_input_grid(input_text)
                #load_list = parse_load(input_text)

                cycles_loss = torch.tensor(10.0)
                invalid_loss = torch.tensor(len(switches))

                # 计算总负荷作为unsupply_loss
                total_load =  parse_total_load(input_text)
                unsupply_loss = torch.tensor(total_load)

                self.loss_scaling_factor[1]=1
                self.loss_scaling_factor[2]=1
            else:
                # 正常解析output中的action
                actions, len_action = parse_action_switch(output_text)
                self.loss_scaling_factor[2]=1/len_action

                # 解析输入网格
                lines, vsources, switches = parse_input_grid(input_text)

                # 构建初始拓扑边
                topo_edges = []
                switch_status = {}

                # 先记录所有开关的初始状态
                for switch in switches:
                    switch_match = re.search(
                        rf"(?i){switch}.*?bus1=([\w]+).*?bus2=([\w]+).*?status=([01])",
                        input_text,
                    )
                    if switch_match:
                        bus1, bus2, status = (
                            switch_match.group(1).upper(),
                            switch_match.group(2).upper(),
                            int(switch_match.group(3)),
                        )
                        switch_status[switch] = (bus1, bus2, status)
                        if status == 1:
                            topo_edges.append((bus1, bus2))

                # 添加非开关线路(默认闭合)
                for line in lines:
                    if line not in switches:
                        line_match = re.search(
                            rf"(?i){line}.*?bus1=([\w]+).*?bus2=([\w]+)",
                            input_text,
                        )
                        if line_match:
                            topo_edges.append(
                                (
                                    line_match.group(1).upper(),
                                    line_match.group(2).upper(),
                                )
                            )

                # 应用动作改变拓扑
                for switch_name, new_status in actions:
                    if switch_name in switch_status:
                        bus1, bus2, _ = switch_status[switch_name]
                        if new_status == 1 and (bus1, bus2) not in topo_edges:
                            topo_edges.append((bus1, bus2))
                        elif new_status == 0 and (bus1, bus2) in topo_edges:
                            topo_edges.remove((bus1, bus2))

                # 计算各项损失
                if "CYL" in self.custom_loss_config:
                    cycles_loss = compute_dis_cycles_loss(topo_edges)
                else:
                    cycles_loss = torch.tensor(0.0)
                if "SUL" in self.custom_loss_config:
                    unsupply_loss = compute_dis_unsupply_loss(
                        input_text, topo_edges, vsources
                    )
                    self.loss_scaling_factor[2] = 1 / parse_total_load(input_text)
                else:
                    self.loss_scaling_factor[2] = 0
                    unsupply_loss = torch.tensor(0.0)
                if "IEL" in self.custom_loss_config:
                    invalid_loss = compute_dis_invalid_loss(input_text, actions)
                else:
                    invalid_loss = torch.tensor(0.0)

        except Exception as e:
            print(f"Error in custom loss calculation: {str(e)}")
            # 如果解析出错，使用最大惩罚值
            lines, vsources, switches = parse_input_grid(input_text)
            cycles_loss = torch.tensor(10.0)
            invalid_loss = torch.tensor(len(switches))
            self.loss_scaling_factor[1]=1
            self.loss_scaling_factor[2]=1

            # 计算总负荷
            total_load = parse_total_load(input_text)
            unsupply_loss = torch.tensor(total_load)

        # 使用loss_scaling_factor进行加权组合损失
        total_loss = loss + (
            cycles_loss * self.loss_scaling_factor[0]
            + unsupply_loss * self.loss_scaling_factor[1]
            + invalid_loss * self.loss_scaling_factor[2]
        )

        # 可选: 打印各项损失值用于调试
        if self.state.global_step % 5 == 0:  # 每5步打印一次
            print(f"\nStep {self.state.global_step} - Loss breakdown:")
            print(f"  Base loss: {loss.item():.4f}")
            print(
                f"  Cycles loss: {cycles_loss.item():.4f} (weight: {self.loss_scaling_factor[0]})"
            )
            print(
                f"  Unsupply loss: {unsupply_loss.item():.4f} (weight: {self.loss_scaling_factor[1]})"
            )
            print(
                f"  Invalid loss: {invalid_loss.item():.4f} (weight: {self.loss_scaling_factor[2]})"
            )
            print(f"  Total loss: {total_loss.item():.4f}\n")

        return (total_loss, outputs) if return_outputs else total_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with specific configurations.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the datasets dir.')
    parser.add_argument('--case_name', type=str, required=True, help='dataset name.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use.')
    parser.add_argument('--output_model', type=str, required=True, help='Output model path.')
    parser.add_argument('--num_train_epochs', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size per device.')
    parser.add_argument('--max_new_tokens', type=int, required=True, help='Maximum number of new tokens generated.')
    # parser.add_argument('--model_name_hf', type=str, required=True, help='Huggingface model name for upload.')
    # parser.add_argument('--tokenizer_name_hf', type=str, required=True, help='Huggingface tokenizer name for upload.')
    parser.add_argument('--custom_loss', type=int, required=True, help='For regular loss use 0. For custom loss with 3 penalty terms use 1.')
    parser.add_argument('--custom_loss_config', type=str, required=True, help='Custom loss configuration.')
    parser.add_argument('--cycles_loss_scaling_factor', type=float, required=True, help='Scaling factor for cycles loss')
    parser.add_argument(
        "--model_for_next_generation_path",
        type=str,
        required=False,
        help="Model path to use for next generation tune.",
    )
    # parser.add_argument(
    #     "--huggingface_token",
    #     type=str,
    #     required=False,
    #     help="Access token of Huggingface hub",
    # )
    return parser.parse_args()

def main():
    args = parse_args()

    print('Training configurations: \n', args)

    # access_token = args.huggingface_token
    # login(token=access_token)

    os.getcwd()

    data_path = args.data_path
    case_name = args.case_name
    model_id = args.model_id
    output_model = args.output_model

    train_dataset, train_df, test_dataset = prepare_resupply_data_llama31(
        data_path, case_name
    )
    print(train_dataset)

    # 加载基础模型和tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id)

    # lora参数
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 检查是否需要加载已有的LoRA模型
    # if args.model_for_next_generation_path:
    #     tune_model_path = args.model_for_next_generation_path
    #     print(f"Loading Peft model from: {tune_model_path}")
    #     lora_model = PeftModel.from_pretrained(
    #         model,
    #         tune_model_path,
    #         config=peft_config,
    #     )
    #     lora_model.train(True)
    #     model=lora_model
    # else:
    #     print("No pre-trained Peft model provided. Training from scratch.")
    #这部分不去判断了，和之前初始化相同

    get_memory_allocated_cached()

    # 配置参数
    training_arguments = SFTConfig(
        output_dir=output_model,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        report_to="none",
        gradient_checkpointing=True,
        dataset_text_field="text",
        packing=False,
        max_seq_length=8192,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        #eval_strategy="epoch",  # 添加评估策略，每个 epoch 进行评估
    )

    if args.custom_loss == 0:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            args=training_arguments,
            processing_class=tokenizer,
            #eval_dataset=test_dataset, 
        )
    elif args.custom_loss == 1:
        trainer = CustomTrainerllama31(
            train_df=train_df,
            cycles_loss_scaling_factor=args.cycles_loss_scaling_factor,
            custom_loss_config=args.custom_loss_config,
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            args=training_arguments,
            processing_class=tokenizer,
            #eval_dataset=test_dataset,
        )

    get_memory_allocated_cached()

    get_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    get_gpu_memory()

    get_nvidia_smi_info(0)

    # 开始训练
    #trainer.train(resume_from_checkpoint=True) 
    #trainer.train(resume_from_checkpoint=args.model_for_next_generation_path) 
    # 添加一个resume_from_checkpoint参数，True表示从最后的检查点恢复训练

    if args.model_for_next_generation_path:
        tune_model_path = args.model_for_next_generation_path
        print(f"Loading Peft model from: {tune_model_path}")
        # 添加一个resume_from_checkpoint参数，表示从指定的检查点恢复训练
        trainer.train(resume_from_checkpoint=tune_model_path)
    else:
        trainer.train(resume_from_checkpoint=False)
        print("No pre-trained Peft model provided. Training from scratch.")

    # 训练完成提示
    print("\nTuning completed successfully!")
    print(f"Model saved to: {output_model}")

if __name__ == "__main__":
    main()
