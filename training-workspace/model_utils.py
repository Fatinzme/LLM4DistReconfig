import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import re
import ast
import networkx as nx
from typing import List, Tuple
from dataset_utils import (
    parse_action_switch,
    parse_circuit_name,
    parse_input_grid,
    parse_load
)


def get_model_and_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        local_files_only=True,
        quantization_config=bnb_config, 
        device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

def get_model(model_id):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        local_files_only=True,
        # device_map=dev_map,
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model

def get_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def parse_open_lines(output_string):
    """
    Parses the open lines from the output string.

    Parameters:
    - output_string: The string containing the open lines information.

    Returns:
    A list of tuples representing the open lines.
    """
    # Regular expression to find the "Open Lines" part
    open_lines_pattern = r'Open Lines=\[(.*?)\]'

    # Search for the pattern in the string
    match = re.search(open_lines_pattern, output_string)

    if match:
        open_lines_str = match.group(1)
        
        # Remove any extraneous characters and ensure proper format
        open_lines_str = re.sub(r'[^0-9,() ]', '', open_lines_str)  # Remove invalid characters
        open_lines_str = re.sub(r'\s+', '', open_lines_str)  # Remove extra whitespace
        open_lines_str = open_lines_str.strip(',')  # Remove trailing commas

        # Ensure that open_lines_str is properly formatted for evaluation
        try:
            open_lines = ast.literal_eval(f'[{open_lines_str}]')
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing open lines: {e}")
            return []

        return open_lines
    else:
        return []


def parse_available_lines(output_string):
        # Regular expression to find the "Lines" part in the prompt which represents the available lines
        open_lines_pattern = r'Lines=\[(.*?)\]'
        
        # Search for the pattern in the string
        match = re.search(open_lines_pattern, output_string)
        
        if match:
            available_lines_str = match.group(1)
            # Convert the string representation of the list to an actual list
            available_lines = ast.literal_eval(f'[{available_lines_str}]')
            return available_lines
        else:
            return []


def get_output_graph_edges(predicted_lines, available_lines):
    predicted_lines_reverse = reverseTuple(predicted_lines)
    predicted_lines.extend(predicted_lines_reverse)
    return [line for line in available_lines if line not in predicted_lines]            


def compute_invalid_edges_loss(predicted_lines, available_lines):
        # Calculate the loss for invalid edges
        invalid_edges_loss = torch.tensor(0.0)
        for line in predicted_lines:
            if line not in available_lines:
                invalid_edges_loss += 1.0  # Adjust the penalty as needed
        return invalid_edges_loss

def compute_cycles_loss(predicted_lines):
    # Calculate the loss for cycles
    G = nx.Graph()
    G.add_edges_from(predicted_lines)
    cycles_loss = torch.tensor(0.0)
    try:
        cycles = list(nx.find_cycle(G, orientation="ignore"))
        cycles_loss += len(cycles)  # Adjust the penalty as needed
    except nx.NetworkXNoCycle:
        pass
    return cycles_loss


def compute_subgraphs_loss(predicted_lines):
    # Calculate the loss for subgraphs
    G = nx.Graph()
    G.add_edges_from(predicted_lines)
    subgraphs_loss = torch.tensor(0.0)
    print('Number of connected components: ', nx.number_connected_components(G)) #remove when training properly
    if nx.number_connected_components(G) > 1:
        subgraphs_loss += nx.number_connected_components(G) - 1  # Adjust the penalty as needed
    return subgraphs_loss    

def reverseTuple(lstOfTuple):
    return [tup[::-1] for tup in lstOfTuple]


def compute_dis_cycles_loss(topo_edges: List[Tuple[str, str]]) -> torch.Tensor:
    """
    计算配电网中的环网数量作为cycles loss

    Args:
        topo_edges: 网络拓扑边列表[(bus1, bus2),...]

    Returns:
        torch.Tensor: 环网数量的损失值
    """
    G = nx.Graph()
    G.add_edges_from(topo_edges)

    try:
        cycles = list(nx.cycle_basis(G))
        return torch.tensor(float(len(cycles)), dtype=torch.float32)
    except nx.NetworkXNoCycle:
        return torch.tensor(0.0, dtype=torch.float32)

def parse_total_load(input_text):
    load_list = parse_load(input_text)
    total_load = 0.0
    for load_name, _ in load_list:
        load_match = re.search(
            rf"(?i)load\.{load_name.split('.')[-1]}.*?P=([\d.]+)",
            input_text,
        )
        if load_match:
            total_load += float(load_match.group(1))
    return total_load
    
    
def compute_dis_unsupply_loss(
    input_text: str, topo_edges: List[Tuple[str, str]], vsources: List[Tuple[str, str]]
) -> torch.Tensor:
    """
    计算停电负荷总量作为unsupply loss

    Args:
        input_text: 输入文本
        topo_edges: 网络拓扑边列表
        vsources: 电源列表[(vsource_name, bus_name),...]

    Returns:
        torch.Tensor: 停电负荷总量的损失值
    """
    # 使用现有的parse_load函数解析负荷
    load_list = parse_load(input_text)

    G = nx.Graph()
    G.add_edges_from(topo_edges)

    # 提取所有连接电源的母线
    source_buses = {bus.upper() for _, bus in vsources}

    # 找出所有没有电源的连通分量
    unsupplied_load = 0.0
    for component in nx.connected_components(G):
        component_has_power = any(bus in source_buses for bus in component)
        if not component_has_power:
            # 找出这个分量中的所有负荷
            for load_name, load_bus in load_list:
                if load_bus.upper() in component:
                    # 从输入文本中提取该负荷的P值
                    load_match = re.search(
                        rf"(?i)load\.{load_name.split('.')[-1]}.*?P=([\d.]+)",
                        input_text,
                    )
                    if load_match:
                        unsupplied_load += float(load_match.group(1))

    return torch.tensor(unsupplied_load, dtype=torch.float32)


def compute_dis_invalid_loss(
    input_text: str, actions: List[Tuple[str, int]]
) -> torch.Tensor:
    """
    计算无效开关动作数量作为invalid loss

    Args:
        input_text: 输入文本
        actions: 开关动作列表[(switch_name, status),...]

    Returns:
        torch.Tensor: 无效动作数量的损失值
    """
    # 使用现有的parse_input_grid函数解析开关信息
    _, _, switches = parse_input_grid(input_text)

    invalid_count = 0
    for switch_name, _ in actions:
        if switch_name not in switches:
            invalid_count += 1

    return torch.tensor(float(invalid_count), dtype=torch.float32)
