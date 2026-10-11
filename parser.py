import torch
import re
import numpy as np

def parse_snns_net(filepath, input_dim):
    """
    Parses an SNNS .net file and maps it to Exact2011Model tensors.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Extract Biases (SNNS "Thresholds" in the Unit Definition section)
    # SNNS format often looks like: no. | typeName | unitName | act | bias | st | pos
    # This regex looks for floating point numbers in the bias column.
    unit_pattern = re.compile(r'^\s*(\d+)\s+\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*([+-]?\d*\.\d+[eE]?[+-]?\d*|\d+)', re.MULTILINE)
    units = unit_pattern.findall(content)
    
    biases = {}
    for unit_id, bias_val in units:
        biases[int(unit_id)] = float(bias_val)

    # 2. Extract Weights (SNNS Connection Definition section)
    # Format typically: target_unit | source_unit:weight, source_unit:weight...
    conn_pattern = re.compile(r'^\s*(\d+)\s*\|(.*)', re.MULTILINE)
    connections = conn_pattern.findall(content[content.find('connection definition'):])

    weights = {}
    for target_id, sources_str in connections:
        target_id = int(target_id)
        weights[target_id] = {}
        # Parse comma-separated source:weight pairs
        source_pairs = re.findall(r'(\d+)\s*:\s*([+-]?\d*\.\d+[eE]?[+-]?\d*|\d+)', sources_str)
        for src_id, weight_val in source_pairs:
            weights[target_id][int(src_id)] = float(weight_val)

    # 3. Map to PyTorch Tensors
    # Assuming standard sequential numbering: Inputs (1 to input_dim), Hidden (input_dim+1 to input_dim+4), Output (input_dim+5)
    hidden_start = input_dim + 1
    output_id = input_dim + 5

    # Initialize PyTorch tensors
    hidden_weight = torch.zeros((4, input_dim), dtype=torch.float32)
    hidden_bias = torch.zeros(4, dtype=torch.float32)
    output_weight = torch.zeros((1, 4), dtype=torch.float32)
    output_bias = torch.zeros(1, dtype=torch.float32)

    # Populate Hidden Layer Tensors
    for h_idx in range(4):
        snns_node_id = hidden_start + h_idx
        hidden_bias[h_idx] = biases.get(snns_node_id, 0.0)
        for i_idx in range(input_dim):
            input_node_id = i_idx + 1 
            hidden_weight[h_idx, i_idx] = weights.get(snns_node_id, {}).get(input_node_id, 0.0)

    # Populate Output Layer Tensors
    output_bias[0] = biases.get(output_id, 0.0)
    for h_idx in range(4):
        snns_node_id = hidden_start + h_idx
        output_weight[0, h_idx] = weights.get(output_id, {}).get(snns_node_id, 0.0)

    # 4. Construct State Dictionary
    state_dict = {
        'input_dim': input_dim,
        'hidden.weight': hidden_weight,
        'hidden.bias': hidden_bias,
        'output.weight': output_weight,
        'output.bias': output_bias
    }
    
    return state_dict

def compile_ensemble(file_mapping, output_path="ppip_ensemble_weights.pt"):
    """
    Loops through all 24 SNNS files and packages them into a single .pt dictionary.
    """
    ensemble_dict = {}
    for combo_key, (filepath, input_dim) in file_mapping.items():
        print(f"Parsing {combo_key} from {filepath}...")
        ensemble_dict[combo_key] = parse_snns_net(filepath, input_dim)
        
    torch.save(ensemble_dict, output_path)
    print(f"Ensemble successfully saved to {output_path}")

# Example Usage:
# file_mapping = {
#     "1_0": ("legacy_nets/model_1_0.net", 160), # Example key and input dimension
#     "1_1": ("legacy_nets/model_1_1.net", 240)
# }
# compile_ensemble(file_mapping)
