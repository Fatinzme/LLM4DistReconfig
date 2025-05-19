"""
保存peft模型
"""
import torch
import argparse
from model_utils import *
from generation_utils import peft_merge_unload

def save_peft_model(model_id, peft_path,save_path):
    model, tokenizer = get_model_and_tokenizer(model_id)
    model_with_peft = peft_merge_unload(model_id, peft_path)
    print(model_with_peft)

    model_with_peft.save_pretrained(save_path)

    tokenizer.save_pretrained(save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a language model with specific configurations to a pre-trained model."
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use.")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Output model path."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Peft of specific model for saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_peft_model(args.model_id, args.peft_path, args.save_path)

if __name__ == "__main__":
    main()
