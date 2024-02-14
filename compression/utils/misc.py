# @Author: Bin Duan <bduan2@hawk.iit.edu>


import torch
from typing import Tuple

MAX_PARAM_SHAPE = 2**16 # maximum number of elements in a tensor
MAX_NN_BYTES = 2**16 # maximum number of bytes in a neural network layer
MAX_HASH_LEVEL_BYTES = 2**24 # maximum number of bytes in a hash level
QUANT_BITDEPTH = 4 # QUANT_BITDEPTH quantization
AC_MAX_VAL = 2 ** QUANT_BITDEPTH - 1 # maximum value for signed QUANT_BITDEPTH quantization

Q_PROBA_DEFAULT = 128.0


_ENCODING_TYPE = [
    "Identity",
    "Frequency",
    "OneBlob",
    "SphericalHarmonics",
    "TriangleWave",
    "Grid",
]
_GRID_TYPE = ["Hash", "Tiled", "Dense"]
_INTERPOLATION_TYPE = ["Nearest", "Linear", "Smoothstep"]
_ACTIVATION_TYPE = [
    "None",
    "ReLU",
    "LeakyReLU",
    "Exponential",
    "Sine",
    "Sigmoid",
    "Squareplus",
    "Softplus",
    "Tanh",
]
_NETWORK_TYPE = ["FullyFusedMLP", "CutlassMLP"]

_TEMPLATE_ENCODING_CONFIG = {
    "Identity": {
        "otype": "Identity",
        "scale": float,  # Scaling of each encoded dimension.
        "offset": float,  # Added to each encoded dimension.
    },
    "Frequency": {
        "otype": "Frequency",
        "n_frequencies": int,  # Number of frequencies (sin & cos) per encoded dimension.
    },
    "OneBlob": {
        "otype": "OneBlob",
        "n_bins": int,
    },  # Number of bins per encoded dimension.
    "SphericalHarmonics": {
        "otype": "SphericalHarmonics",
        "degree": int,  # The SH degree up to which to evaluate the encoding. Produces degree^2 encoded dimensions.
    },
    "TriangleWave": {
        "otype": "TriangleWave",
        "n_frequencies": int,  # Number of frequencies (triwave) per encoded dimension.
    },
    "Grid": {
        "otype": "Grid",
        "type": int,  # Type of backing storage of the grids. Can be "Hash", "Tiled" or "Dense".
        "n_levels": int,  # Number of levels (resolutions)
        "n_features_per_level": int,  # Dimensionality of feature vector stored in each level"s entries.
        "log2_hashmap_size": int,  # If type is "Hash", is the base-2 logarithm of the number of elements in each backing hash table.
        "base_resolution": int,  # The resolution of the coarsest level is base_resolution^input_dims.
        "per_level_scale": float,  # The geometric growth factor, i.e. the factor by which the resolution of each grid is larger (per axis) than that of the preceding level.
        "interpolation": int,  # How to interpolate nearby grid lookups. Can be "Nearest", "Linear", or "Smoothstep" (for smooth derivatives).
    },
}

_TEMPLATE_NETWORK_CONFIG = {
    "otype": str,  # component type
    "activation": int,  # Activation of hidden layers.
    "output_activation": int,  # Activation of the output layer.
    "n_neurons": int,  # Neurons in each hidden layer. May only be 16, 32, 64, or 128 for fully_fused MLP
    "n_hidden_layers": int,  # Number of hidden layers.
}


#### functions ####
def generate_input_grid(
    resolution: Tuple[int, int, int], # height, width
    device: str,
):
    # Variables for saving/displaying image results

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys], indexing="ij")

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    return xy


def get_num_layers(config: dict) -> int:
    """Get the number of layers in the network.

    Args:
        config (dict): The configuration file

    Returns:
        int: The number of layers in the network
    """
    return config["n_hidden_layers"] + 1


def get_num_levels(config: dict) -> int:
    """Get the number of levels in the input encoding.

    Args:
        config (dict): The configuration file

    Returns:
        int: The number of levels in the input encoding
    """
    otype = config["otype"]
    num_levels = 1

    if otype == "Identity":
        num_levels = 1
    elif otype == "Frequency":
        num_levels = config["n_frequencies"]
    elif otype == "OneBlob":
        num_levels = config["n_bins"]
    elif otype == "SphericalHarmonics":
        num_levels = 1
    elif otype == "TriangleWave":
        num_levels = config["n_frequencies"]
    elif otype == "Grid":
        num_levels = config["n_levels"]

    return num_levels

def generate_param_index_list(elements_count_list):
    # Calculate cumulative sum
    cumulative_sum = [0] + elements_count_list
    for i in range(1, len(cumulative_sum)):
        cumulative_sum[i] += cumulative_sum[i - 1]

    # Generate indices list
    indices_list = [cumulative_sum[i] for i in range(1, len(cumulative_sum))]

    return indices_list

def reverse_generate_param_index_list(indices_list):
    # Calculate the differences between consecutive indices
    counts_list = [indices_list[0]]
    for i in range(1, len(indices_list)):
        counts_list.append(indices_list[i] - indices_list[i - 1])

    return counts_list

def get_param_partitions(encoding_config: dict, network_config: dict, n_output_dims: int) -> list[int]:
    # get partitions for input encoding
    config = encoding_config
    otype = config["otype"]
    if otype == "Grid":
        # Calculate resolutions for each level
        resolutions = torch.ceil(2 ** (torch.arange(config["n_levels"]) * torch.log2(torch.tensor(config["per_level_scale"]))) * config["base_resolution"] - 1) + 1

        # Calculate the number of parameters for each level
        params_per_level = (resolutions ** 2 + 7) // 8 * 8 # to align with memory

        # Truncate parameters if they exceed the size of the hashmap
        params_per_level = torch.minimum(params_per_level, torch.tensor(2 ** config["log2_hashmap_size"])).int() * config["n_features_per_level"]

    config = network_config
    padded_output_width = (n_output_dims + 15) // 16 * 16 # to align with memory
    padded_input_width = (encoding_config["n_levels"] * encoding_config["n_features_per_level"] + 3) // 4 * 4 # to align with memory

    params_per_layer = [padded_input_width * config["n_neurons"]] + [config["n_neurons"] * config["n_neurons"] for _ in range(config["n_hidden_layers"] - 1)] + [config["n_neurons"] * padded_output_width]
    
    elements_count_list = params_per_level.tolist() + params_per_layer
    return generate_param_index_list(elements_count_list)
        