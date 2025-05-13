import argparse
import networkx as nx
import sys
import os
import torch
from time import perf_counter

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
    parser.add_argument('--model_for_generation_path', type=str, required=True, help='Model path to use for generation.')
    parser.add_argument('--response_to', type=str, required=False, help='directory to save the response.')
    return parser.parse_args()

def main():
    args = parse_args()

    train_dataset, train_df,test_dataset = prepare_resupply_data_llama31(
        args.data_path,args.case_name
    )

    model_id = args.model_id

    model, tokenizer = get_model_and_tokenizer(model_id)

    get_memory_allocated_cached()

    os.getcwd()

    model_path = args.model_for_generation_path

    model_with_peft = peft_merge_unload(model_id, model_path)

    print(model_with_peft)

    p1,p2=generate_response(user_input=train_df['input'].iloc[0], model=model_with_peft, tokenizer =tokenizer, max_new_tokens=8192)
    #在response_to文件夹下保存p1
    if args.response_to:
        os.makedirs(args.response_to, exist_ok=True)
        with open(os.path.join(args.response_to, 'response1.txt'), 'w') as f:
            f.write(p1)

    p1,p2=generate_response(user_input=train_df['input'].iloc[1], model=model_with_peft, tokenizer =tokenizer, max_new_tokens=8192)
    #在response_to文件夹下保存p1
    if args.response_to:
        os.makedirs(args.response_to, exist_ok=True)
        with open(os.path.join(args.response_to, 'response2.txt'), 'w') as f:
            f.write(p1)


if __name__ == "__main__":
    main()
