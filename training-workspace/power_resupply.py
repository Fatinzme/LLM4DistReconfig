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
from trl import SFTTrainer
import subprocess as sp
from time import perf_counter
import nvidia_smi

# sys.path.append(os.path.abspath('/home/ubuntu/LLMCODE/LLM4DistReconfig/Dataset-Notebooks/utils'))
from dataset_utils import *
from model_utils import *
from generation_utils import *
from monitoring_utils import *


class CustomTrainerllama31(SFTTrainer):

    def __init__(self, train_df=None, cycles_loss_scaling_factor=None, * args, **kwargs):
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
        if loss_scaling_factor is None:
            self.loss_scaling_factor = [1.0, 0.01, 1.0]  # 默认权重
        else:
            self.loss_scaling_factor = [cycles_loss_scaling_factor, 0.01, 1.0]



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
        cycles_loss = torch.tensor(0.0).to(loss.device)
        unsupply_loss = torch.tensor(0.0).to(loss.device)
        invalid_loss = torch.tensor(0.0).to(loss.device)

        # 检查输出是否成功
        success_match = re.search(
            r'"success"\s*:\s*(true|false)', output_text, re.IGNORECASE
        )

        try:
            # 如果output解析失败或success=false，使用惩罚值
            if not success_match or success_match.group(1).lower() == "false":
                lines, vsources, switches = parse_input_grid(input_text)
                load_list = parse_load(input_text)

                cycles_loss = torch.tensor(10.0).to(loss.device)
                invalid_loss = torch.tensor(len(switches)).to(loss.device)

                # 计算总负荷作为unsupply_loss
                total_load = 0.0
                for load_name, _ in load_list:
                    load_match = re.search(
                        rf"(?i)new\s+load\.{load_name.split('.')[-1]}.*?P=([\d.]+)",
                        input_text,
                    )
                    if load_match:
                        total_load += float(load_match.group(1))
                unsupply_loss = torch.tensor(total_load).to(loss.device)
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
                        rf"(?i)new\s+{switch}.*?bus1=([\w]+).*?bus2=([\w]+).*?status=([01])",
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
                            rf"(?i)new\s+{line}.*?bus1=([\w]+).*?bus2=([\w]+)",
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
                cycles_loss = compute_dis_cycles_loss(topo_edges)
                unsupply_loss = compute_dis_unsupply_loss(
                    input_text, topo_edges, vsources
                )
                invalid_loss = compute_dis_invalid_loss(input_text, actions)

                self.loss_scaling_factor[2]=1/parse_total_load(input_text)

        except Exception as e:
            print(f"Error in custom loss calculation: {str(e)}")
            # 如果解析出错，使用最大惩罚值
            lines, vsources, switches = parse_input_grid(input_text)
            cycles_loss = torch.tensor(10.0).to(loss.device)
            invalid_loss = torch.tensor(len(switches)).to(loss.device)
            self.loss_scaling_factor[1]=1
            self.loss_scaling_factor[2]=1

            # 计算总负荷
            total_load = parse_total_load(input_text)
            unsupply_loss = torch.tensor(total_load).to(loss.device)

        # 使用loss_scaling_factor进行加权组合损失
        total_loss = loss + (
            cycles_loss * self.loss_scaling_factor[0]
            + unsupply_loss * self.loss_scaling_factor[1]
            + invalid_loss * self.loss_scaling_factor[2]
        )

        # 可选: 打印各项损失值用于调试
        if self.state.global_step % 100 == 0:  # 每100步打印一次
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
    parser.add_argument('--model_for_generation_path', type=str, required=True, help='Model path to use for generation.')
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

    train_dataset, train_df = prepare_resupply_data_llama31(
        data_path,case_name
    )
    print(train_dataset)

    model, tokenizer = get_model_and_tokenizer(model_id)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # if torch.cuda.device_count()  > 1:
    #     model = torch.nn.DataParallel(model)

    get_memory_allocated_cached()

    peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )

    # 可能要把默认
    training_arguments = TrainingArguments(
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
##############################################

        )

    if args.custom_loss==0:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            #dataset_text_field="text",
            args=training_arguments,
            processing_class=tokenizer,
            # packing=False,
            # max_seq_length=4096
            
        )

    elif args.custom_loss==1:
        trainer = CustomTrainerllama31(
            train_df=train_df,
            cycles_loss_scaling_factor=args.cycles_loss_scaling_factor,
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            args=training_arguments,
            processing_class=tokenizer,
            # packing=False,
            # max_seq_length=4096,
            # dataset_text_field="text",
            # 存在问题，需要查看源码
        )

    get_memory_allocated_cached()

    get_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device) 

    get_gpu_memory()

    get_nvidia_smi_info(0)

    trainer.train()

    model_path = args.model_for_generation_path

    model_with_peft = peft_merge_unload(model_id, model_path)

    print(model_with_peft)

    # model_with_peft.module.save_pretrained("./lora_finetuned_model") if torch.cuda.device_count() > 1 else model_with_peft.save_pretrained("./lora_finetuned_model")
    # tokenizer.save_pretrained("./lora_finetuned_model")

    # model_with_peft.push_to_hub(args.model_name_hf, access_token)
    # tokenizer.push_to_hub(args.tokenizer_name_hf, access_token)

    generate_response(user_input=train_dataset['input'][0], model=model_with_peft, tokenizer =tokenizer)

if __name__ == "__main__":
    main()
