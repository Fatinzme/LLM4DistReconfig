#!/bin/bash


# accelerate launch --config_file /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml /home/ubuntu/LLMCODE/LLM4DistReconfig/Model-Notebooks/llama3-notebooks/grid-reconfiguration.py \

# python /home/ubuntu/LLMCODE/LLM4DistReconfig/Model-Notebooks/llama3-notebooks/test.py \
python /home/ubuntu/LLMCODE/LLM4DistReconfig/Model-Notebooks/llama3-notebooks/grid-reconfiguration.py \
    --data_path /home/ubuntu/LLMCODE/LLM4DistReconfig_Data/pre_data/train_33_69_84_nodes.csv \
    --model_id /home/ubuntu/LLMCODE/LLM4DistReconfig/Meta-Llama-3___1-8B-Instruct \
    --output_model /home/ubuntu/LLMCODE/LLM4DistReconfig/LoRA/checkpoint-3285 \
    --num_train_epochs 1 \
    --batch_size 4 \
    --custom_loss 1 \
    --custom_loss_config IEL,SUL,CYL \
    --cycles_loss_scaling_factor 1 \
    --model_for_generation_path /home/ubuntu/LLMCODE/LLM4DistReconfig/LoRA/checkpoint-3285 \
    --max_new_tokens 1200 

    #--model_name_hf model_name
    #--tokenizer_name_hf tokenizer_name