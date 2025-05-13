#!/bin/bash

python  /media/ubuntu/storage/LLMSupplyRestore/LLM4DistReconfig/training-workspace/power_resupply.py \
            --data_path /media/ubuntu/storage/LLMSupplyRestore/LLM4DistReconfig/Datasets  \
            --case_name Taiwan96  \
            --model_id "/media/ubuntu/storage/LLMSupplyRestore/pre-trained models/Meta-Llama-3___1-8B-Instruct"  \
            --output_model /media/ubuntu/storage/LLMSupplyRestore/TUNE_DIS_LLM/LoRA/checkpoint-3285  \
            --num_train_epochs 2  \
            --batch_size 4  \
            --custom_loss 1  \
            --custom_loss_config IEL,SUL,CYL  \
            --cycles_loss_scaling_factor 1  \
            --model_for_generation_path /media/ubuntu/storage/LLMSupplyRestore/TUNE_DIS_LLM/LoRA/checkpoint-3285  \
            --max_new_tokens 3000

    #--model_name_hf model_name
    #--tokenizer_name_hf tokenizer_name