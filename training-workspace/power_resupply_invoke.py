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

def parse_args():
    parser = argparse.ArgumentParser(description="Use a language model with specific configurations.")
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
    return parser.parse_args()

def main():
    args = parse_args()

    os.getcwd()

    model_id = args.model_id

    model_path = args.model_for_generation_path

    model_with_peft = peft_merge_unload(model_id, model_path)

    print(model_with_peft)

    generate_response(user_input=train_dataset['input'][0], model=model_with_peft, tokenizer =tokenizer)

if __name__ == "__main__":
    main()
