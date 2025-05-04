import argparse
import pandas as pd
from pathlib import Path

from dataset_utils import *
from model_utils import *

def test_pipeline():
    # 1. 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径')
    parser.add_argument('--n_rows', type=int, default=10, help='输出的行数(默认为10)')
    parser.add_argument('--case_name', type=str, required=True, help='要读取的算例名称')
    args = parser.parse_args()


    # 创建输出目录(如果不存在)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 2. 调用数据准备函数
    print("\n======== 开始数据准备 ========")
    train_dataset, train_df = prepare_resupply_data_llama31(args.data_path, args.case_name)
    print(f"数据加载完成，共 {len(train_df)} 个样本")
    
    # 3. 输出初始数据
    output_file1 = output_dir / "train_df1.xlsx"
    train_df.head(args.n_rows).to_excel(output_file1)
    print(f"\n初始数据前{args.n_rows}行已保存到: {output_file1}")
    # 4. 调用输入输出解析函数
    print("\n======== 开始解析开关动作 ========")
    train_df_action, train_df_with_action = parse_input_output(train_df)
    print("开关动作解析完成")
    # 5. 输出解析后的数据
    output_file2 = output_dir / "train_df2.xlsx"
    train_df_with_action.head(args.n_rows).to_excel(output_file2)
    print(f"\n带动作次数的数据前{args.n_rows}行已保存到: {output_file2}")
    # 6. 输出开关动作信息
    output_file3 = output_dir / "train_df_action.txt"
    selected_names = train_df.index[:args.n_rows]
    
    with open(output_file3, 'w', encoding='utf-8') as f:
        f.write("========== 开关动作信息 ==========\n")
        for name in selected_names:
            actions = train_df_action.get(name, [])
            f.write(f"\n电路名称: {name}\n")
            if actions:
                f.write("开关动作列表:\n")
                for switch, status in actions:
                    f.write(f"  - 开关 {switch}: 新状态={status}\n")
            else:
                f.write("无开关动作\n")
    
    print(f"\n开关动作详细信息已保存到: {output_file3}")
    print("\n======== 测试完成 ========")
if __name__ == "__main__":
    test_pipeline()
