import argparse
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull a language model with model-id."
    )
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use.')
    parser.add_argument(
        "--huggingface_token",
        type=str,
        required=False,
        help="Access token of Huggingface hub",
    )
    return parser.parse_args()


def get_model_and_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def main():
    args = parse_args()
    if args.huggingface_token:
        login(token=args.huggingface_token)
        print(f"获取{args.model_id}")
        model, tokenizer = get_model_and_tokenizer(args.model_id)
    else:
        print("请输入Huggingface Hub的Access Token")


if __name__ == "__main__":
    main()
