from datasets import load_dataset, Dataset
import ast
import pickle
import numpy as np
import pandas as pd
import glob as glob
import os
import re
from typing import Dict, Tuple


def clean_value(value):
    # Function to remove square brackets and convert string representation of lists to actual lists
    
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def modify_and_save(file_path, set_value, df_column, output_file_path):
    file_path = file_path
    df = pd.read_csv(file_path)
    specific_value = set_value
    df[df_column] = specific_value
    output_file_path = output_file_path
    df.to_csv(output_file_path, index=False)

def convert_to_dict(data):
    # Convert each row into a dictionary
    data_dict_list = []
    for _, row in data.iterrows():
        entry = {column: clean_value(value) for column, value in row.items()}
        data_dict_list.append(entry)
    return data_dict_list

def dict_to_pickle(data_dict_list, output_file_path):
    # Save the list of dictionaries to a Pickle file
    file_path = output_file_path
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(data_dict_list, pickle_file)
    print(f"Data successfully saved to {output_file_path}")

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_inputs(data):
    # Function to create text inputs
    inputs = []
    for entry in data:
        input = (
            f"Power Distribution Network: Busses={entry['buses']}, Lines={entry['lines']}, "
            f"Line Impedances={entry['line_impedances']}, Open Lines={entry['existing_open_lines']}\n"
            f"Network Variables: NodeVoltages={entry['existing_node_voltages']}, "
            f"System Loss={entry['existing_system_loss']}, System Load={entry['system_load']}\n"
        )
        inputs.append(input)
    return inputs


def create_outputs(data):
    # Function to create text outputs
    outputs = []
    for entry in data:
        output = (
            f"Output: Open Lines={entry['updated_open_lines']}, "
            f"Node Voltages={entry['updated_node_voltages']}, System Loss={entry['updated_system_loss']}\n"
        )
        outputs.append(output)
    return outputs

def save_to_txt(file_path, entries):
    with open(file_path, 'w') as f:
        for entry in entries:
            f.write(entry + '<end>')


def load_entries(file_path):
    # Function to load prompts from the text file
    with open(file_path, 'r') as f:
        entries = f.read().strip().split('<end>')

    entries.pop()
    return entries

def create_task_descriptions(task_description, inputs):
    task_descriptions = [task_description] * len(inputs)
    return task_descriptions

def create_prompts(task_descriptions, inputs):
    prompts = [desc + "\n" + inp for desc, inp in zip(task_descriptions, inputs)]
    return prompts

def create_train_df(task_descriptions, inputs, prompts, outputs):
    train_data = {'Task Description': task_descriptions, 'input': inputs, 'prompt': prompts,'output': outputs}
    train_df = pd.DataFrame(train_data)
    return train_df

def generate_random_split(train_df):
    num_samples = len(train_df)

    # Generate random splits
    np.random.seed(42)  # For reproducibility
    split_values = ['train'] * (num_samples // 3) + ['test'] * (num_samples // 3) + ['validation'] * (num_samples // 3)
    
    # If there are remaining samples, assign them randomly
    remainder = num_samples - len(split_values)
    split_values += np.random.choice(['train', 'test', 'validation'], size=remainder).tolist()
    
    # Shuffle the split assignments
    np.random.shuffle(split_values)
    
    # Add the split column to the DataFrame
    train_df['split'] = split_values

def combine_csv_files(file_paths, output_file_path):
    """
    Combines multiple CSV files into a single DataFrame and resets the index.

    Parameters:
    - file_paths: List of file paths to the CSV files to be combined.
    - output_file_path: Path where the combined CSV file will be saved.
    """
    # Initialize an empty list to store individual DataFrames
    data_frames = []

    # Read each file and append the DataFrame to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Reset the index to ensure unique indices
    combined_df.reset_index(drop=True, inplace=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file_path, index=False)
    print(f"Combined CSV saved to {output_file_path}")

def formatted_train(input,response)->str:
    return f"<|user|>\n{input}</s>\n<|assistant|>\n{response}</s>"


def prepare_train_data(data_path):
    dataset = load_dataset('csv', data_files=data_path)
    print(dataset)
    
    # Filter the dataset for the 'train', 'validation', and 'test' splits
    train_dataset = dataset['train'].filter(lambda x: x['split'] == 'train')
    validation_dataset = dataset['train'].filter(lambda x: x['split'] == 'validation')
    test_dataset = dataset['train'].filter(lambda x: x['split'] == 'test')

    # Print the first entry of each dataset to verify
    print(train_dataset[0]['split'])
    print(validation_dataset[0]['split'])
    print(test_dataset[0]['split'])
    
    # Remove the 'split' column as it's no longer needed
    train_dataset = train_dataset.remove_columns('split')
    validation_dataset = validation_dataset.remove_columns('split')
    test_dataset = test_dataset.remove_columns('split')

    train_df = train_dataset.to_pandas()
    train_df["text"] = train_df[["prompt", "output"]].apply(lambda x: "<|user|>\n" + x["prompt"] + "</s>\n<|assistant|>\n" + x["output"] + "</s>", axis=1)
    train_dataset = Dataset.from_pandas(train_df)

    validation_df = validation_dataset.to_pandas()
    validation_df["text"] = validation_df[["prompt", "output"]].apply(lambda x: "<|user|>\n" + x["prompt"] + "</s>\n<|assistant|>\n" + x["output"] + "</s>", axis=1)
    validation_dataset = Dataset.from_pandas(validation_df)
    
    test_df = test_dataset.to_pandas()
    test_df["text"] = test_df[["prompt", "output"]].apply(lambda x: "<|user|>\n" + x["prompt"] + "</s>\n<|assistant|>\n" + x["output"] + "</s>", axis=1)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, validation_dataset, test_dataset


def parse_action_switch(output_text):
    """
    解析模型输出中的开关动作

    参数:
        output_text (str): 模型输出的文本内容，包含开关动作信息

    返回:
        tuple: (开关动作列表, 动作次数)
            - 开关动作列表: 每个元素为(开关名称, 新状态)的元组
            - 动作次数: 总共有多少个开关动作

    功能说明:
        1. 从输出文本中解析所有开关状态变化
        2. 使用正则表达式匹配形如"Line.XXX.status=Y"的格式
        3. 返回解析结果和动作总次数
        4. 开关名称不区分大小写
    """
    actions = []
    # 纯文本格式的开关动作解析
    text_actions = re.finditer(r"(?i)line\.([\w]+)\.status=([01])", output_text)
    for action in text_actions:
        switch_name = action.group(1).upper()
        new_status = int(action.group(2))
        actions.append((f"Line.{switch_name}", new_status))

    return actions, len(actions)

def parse_input_grid(input_text):
    """
    解析电网输入数据，提取线路、电源和开关信息

    参数:
        input_text (str): 输入电网文本（来自dis_fa*.txt）

    返回:
        tuple: (线路列表, 电源列表, 开关列表)
            - 线路列表: 所有线路名称（Line.*）
            - 电源列表: (电源名称, 所接母线)元组列表
            - 开关列表: 所有可操作开关（type=S的线路）

    功能说明:
        1. 识别所有线路（包括普通线路和开关线路）
        2. 识别所有电源及其连接的母线
        3. 从线路中过滤出可操作的开关（type=S的线路）
        4. 所有名称统一转换为大写格式
    """
    lines = []
    vsources = []
    switches = []

    # 解析所有线路（包括开关线路）
    line_matches = re.finditer(r"(?i)new\s+line\.([\w]+)\s+.*?type=([\w]+)", input_text)

    for match in line_matches:
        line_name = f"Line.{match.group(1).upper()}"  # 统一线路名称格式
        line_type = match.group(2).upper()
        lines.append(line_name)
        if line_type == "S":
            switches.append(line_name)  # 只有type=S的线路才是可操作开关

    # 解析所有电源（VSource）
    vsource_matches = re.finditer(
        r"(?i)new\s+vsource\.([\w]+)\s+bus1=([\w]+)", input_text
    )

    for match in vsource_matches:
        vsource_name = f"VSource.{match.group(1).upper()}"
        bus_name = f"{match.group(2).upper()}"
        vsources.append((vsource_name, bus_name))

    return lines, vsources, switches


def parse_load(input_text):
    """
    解析输入文本中的负荷(Load)及其连接母线信息

    参数:
        input_text (str): 输入电网文本（来自dis_fa*.txt）

    返回:
        list: 包含(负荷名称, 所连母线)元组的列表，所有名称统一为大写格式

    功能说明:
        1. 从input_text中使用正则表达式匹配所有Load定义
        2. 提取负荷名称和连接的母线编号
        3. 自动处理大小写不敏感的情况
        4. 返回格式示例：[("LOAD1", "BUS1"), ("LOAD2", "BUS2")]

    正则表达式说明:
        (?i) - 不区分大小写匹配
        \bnew\s+load\. - 匹配"New Load."或"new load."等变体
        ([\w]+) - 捕获负荷名称(字母数字下划线)
        .*?bus1= - 跳过中间字符直到bus1=
        ([\w]+) - 捕获连接的母线编号
    """
    # 初始化结果列表
    load_list = []

    # 使用正则表达式查找所有Load定义
    load_pattern = r"(?i)\bnew\s+load\.([\w]+).*?bus1=([\w]+)"

    # 查找所有匹配项
    matches = re.finditer(load_pattern, input_text)

    # 处理每个匹配项
    for match in matches:
        load_name = f"Load.{match.group(1).upper()}"  # 统一转换为大写
        bus_name = f"{match.group(2).upper()}"  # 统一转换为大写
        load_list.append((load_name, bus_name))

    return load_list


def parse_circuit_name(input_text):
    """
    从输入文本中解析电路名称

    参数:
        input_text (str): 输入电网文本（来自dis_fa*.txt）

    返回:
        str: 电路名称（大写格式）

    功能说明:
        1. 首先尝试匹配显式的电路名称定义（New Circuit.xxx）
        2. 如果没有找到，则使用第一个线路名称作为电路标识
        3. 确保返回的名称是大写格式
        4. 如果都不能识别，返回"UNKNOWN"
    """
    # 1. 尝试查找显示的电路名称定义
    match = re.search(r"(?i)\bnew\s+circuit\.([\w]+)", input_text)

    if match:
        return match.group(1).upper()

    # 2. 如果没有显式定义，则使用第一个线路的名称
    line_match = re.search(r"(?i)\bnew\s+line\.([\w]+)", input_text)

    return line_match.group(1).upper() if line_match else "UNKNOWN"


def prepare_resupply_data_llama31(data_dir_path, case_name):
    """
    为微调大模型准备数据

    参数:
        data_dir_path (str): 包含任务描述和案例文件夹的目录路径
        case_name (str): 要处理的案例名称(文件夹名),如果None则处理所有案例

    返回:
        tuple: (train_dataset, train_df) 数据集和对应的DataFrame
    """
    # 1. 读取任务描述文件
    task_file = os.path.join(data_dir_path, "task description.txt")
    with open(task_file, "r") as f:
        task_description = f.read().strip()

    # 2. 收集所有要处理的案例
    if case_name:
        case_dirs = [os.path.join(data_dir_path, case_name)]
    else:
        raise ValueError("parameter case_name must be provided.")

    # 3. 初始化DataFrame
    train_data = {"name": [], "input": [], "optm_output": [], "text": []}

    # 4. 处理每个案例文件夹
    for case_dir in case_dirs:
        case_path = os.path.join(data_dir_path, case_dir)

        # 查找所有的dis_fa*.txt文件
        dis_files = [
            f
            for f in os.listdir(case_path)
            if f.startswith("dis_fa") and f.endswith(".txt")
        ]

        for dis_file in dis_files:
            # 构造对应的sol_fa*.txt文件名
            sol_file = dis_file.replace("dis_", "sol_")
            sol_path = os.path.join(case_path, sol_file)

            # 如果对应的解文件存在
            if os.path.exists(sol_path):
                # 读取输入(dis_fa文件)内容
                with open(os.path.join(case_path, dis_file), "r") as f:
                    dis_content = f.read()

                # 读取输出(sol_fa文件)内容
                with open(sol_path, "r") as f:
                    sol_content = f.read()

                # 解析电路名称
                circuit_name = parse_circuit_name(dis_content)

                # 构造样本文本
                sample_text = f"<|user|>\n<task>\n{task_description}\n</task>\n<grid>\n{dis_content}\n</grid>\n</s>\n<|assistant|>\n{sol_content}\n</s>"

                # 添加到数据字典中
                train_data["name"].append(circuit_name)
                train_data["input"].append(dis_content)
                train_data["optm_output"].append(sol_content)
                train_data["text"].append(sample_text)

    # 5. 创建DataFrame并设置index
    train_df = pd.DataFrame(train_data)
    train_df.set_index("name", inplace=True)

    # 6. 转换为HuggingFace数据集
    train_dataset = Dataset.from_pandas(train_df[["text"]])

    return train_dataset, train_df


def parse_action_switch(output_text: str) -> Tuple[list, int]:
    """
    解析模型输出中的开关动作

    参数:
        output_text (str): 模型输出的文本内容，包含开关动作信息

    返回:
        tuple: (开关动作列表, 动作次数)
            - 开关动作列表: 每个元素为(开关名称, 新状态)的元组
            - 动作次数: 总共有多少个开关动作

    功能说明:
        1. 从输出文本中解析所有开关状态变化
        2. 使用正则表达式匹配形如"Line.XXX.status=Y"的格式
        3. 返回解析结果和动作总次数
        4. 开关名称不区分大小写
    """
    # 使用正则表达式找到所有开关动作
    action_matches = re.finditer(r"(?i)line\.([a-z0-9_]+)\.status=([01])", output_text)

    switch_actions = []
    for match in action_matches:
        switch_name = match.group(1)
        new_status = int(match.group(2))
        switch_actions.append((switch_name.upper(), new_status))  # 统一转换为大写

    # 动作次数就是开关动作列表的长度
    return switch_actions, len(switch_actions)


def parse_input_output(train_df: pd.DataFrame) -> Tuple[Dict[str, list], pd.DataFrame]:
    """
    解析模型输入输出，提取开关动作及其次数

    参数:
        train_df (pd.DataFrame): 包含输入输出的DataFrame
            - 要求包含'optm_output'列，存储模型输出文本

    返回:
        tuple: (train_df_action, train_df)
            - train_df_action: 字典，key为电路名称，value为parse_action_switch返回的结果
            - train_df: 新增'act_num'列后的DataFrame

    功能说明:
        1. 遍历train_df的每一行(每个样本)
        2. 对每个样本的optm_output调用parse_action_switch
        3. 将结果存入字典并添加动作次数列
        4. 处理后DataFrame新增'act_num'列记录每个样本的开关动作次数
    """
    # 初始化存储开关动作的字典
    train_df_action = {}

    # 初始化动作次数列表(长度与DataFrame行数相同)
    act_num_list = []

    # 遍历DataFrame的每一行
    for idx, row in train_df.iterrows():
        # 提取当前样本的电路名称(即index)
        circuit_name = idx

        # 获取模型输出文本
        output_text = row["optm_output"]

        # 解析开关动作
        switch_actions, act_num = parse_action_switch(output_text)

        # 存入字典
        train_df_action[circuit_name] = switch_actions

        # 记录动作次数
        act_num_list.append(act_num)

    # 添加act_num列到DataFrame
    modified_df = train_df.copy()
    modified_df["act_num"] = act_num_list

    return train_df_action, modified_df
