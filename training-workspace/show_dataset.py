import argparse
import json
from dataset_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with specific configurations.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    return parser.parse_args()

def main():
    args = parse_args()
    train_dataset, validation_dataset, test_dataset = prepare_train_data(args.data_path)
    with open("./data1.json","w",encoding="utf-8") as fp:
        json.dump(train_dataset[:10],fp,ensure_ascii=False,indent=2)
    with open("./data2.json","w",encoding="utf-8") as fp:   
        json.dump(test_dataset[:10],fp,ensure_ascii=False,indent=2)


if __name__ == "__main__":
    main()
