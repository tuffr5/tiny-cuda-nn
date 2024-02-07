# @Author: Bin Duan <bduan2@hawk.iit.edu>


import torch
from typing import Tuple


MAX_AC_MAX_VAL = 65535 # 2**16 for 16-bit code in bitstream header.
MAX_NN_BYTES = 2**32 - 1 # 2**32 for 32-bit code in bitstream header.
Q_PROBA_DEFAULT = 128.0

POSSIBLE_Q_STEP_NN = 2. ** torch.linspace(-7, 0, 8, device='cpu')

# Avoid numerical instability when measuring the rate of the NN parameters
MIN_SCALE_NN_WEIGHTS_BIAS = 1.0/1024.0

# List of all possible scales when coding a MLP
POSSIBLE_SCALE_NN = 2 ** torch.linspace(MIN_SCALE_NN_WEIGHTS_BIAS, 16, steps=2 ** 16 - 1, device='cpu')

FIXED_POINT_FRACTIONAL_BITS = 6 # 8 works fine in pure int mode (ARMINT True).
                                # reduced to 6 for now for int-in-fp mode (ARMINT False)
                                # that has less headroom (23-bit mantissa, not 32)
FIXED_POINT_FRACTIONAL_MULT = 2 ** FIXED_POINT_FRACTIONAL_BITS

_INPUT_ENCODING_TYPE = [
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

_TEMPLATE_INPUT_ENCODING_CONFIG = {
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