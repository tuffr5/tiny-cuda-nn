# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import math
import os
import subprocess
import torch

from bitstream.header import write_frame_header
from bitstream.range_coder import RangeCoder
from typing import Tuple

from utils.misc import POSSIBLE_Q_STEP_NN, POSSIBLE_SCALE_NN


@torch.no_grad()
def encode_frame(model, bitstream_path: str, img_size: Tuple[int, int, int], config: dict):
    """Convert a model to a bitstream located at <bitstream_path>.

    Args:
        model: A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """

    torch.use_deterministic_algorithms(True)

    model.eval()
    model.to('cpu')

    subprocess.call(f'rm -f {bitstream_path}', shell=True)

    # ================= Encode the MLP into a bitstream file ================ #
    q_step_index_nn = int(torch.argmin((POSSIBLE_Q_STEP_NN - model.get_q_step()).abs()).item())

    # Quantize the weight with the actual quantization step
    model_param_quant = torch.round(model.params / POSSIBLE_Q_STEP_NN[q_step_index_nn])

    # Compute AC_MAX_VAL.
    ac_max_val_nn = int(torch.ceil(model_param_quant.abs().max() + 2).item())
    
    range_coder_nn = RangeCoder(ac_max_val_nn)

    floating_point_scale_weight = model_param_quant.std().item() / math.sqrt(2)

    # Find the closest element to the actual scale in the POSSIBLE_SCALE_NN list
    scale_index_nn = int(torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_weight).abs()).item())

    # ----------------- Actual entropy coding
    cur_bitstream_path = f'{bitstream_path}_MLP'
    range_coder_nn.encode(
        cur_bitstream_path,
        model_param_quant,
        torch.zeros_like(model_param_quant),
        POSSIBLE_SCALE_NN[scale_index_nn] * torch.ones_like(model_param_quant)
    )

    n_bytes_nn = os.path.getsize(cur_bitstream_path)
    # ================= Encode the MLP into a bitstream file ================ #
    
    # Write the header
    header_path = f'{bitstream_path}_header'
    success = write_frame_header(
        header_path,
        img_size,
        config,
        ac_max_val_nn,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
    )

    if success:
        # Concatenate everything inside a single file
        subprocess.call(f'rm -f {bitstream_path}', shell=True)
        subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {header_path}', shell=True)

        subprocess.call(f'cat {cur_bitstream_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {cur_bitstream_path}', shell=True)
    
    else:
        # remove intermediate bitstream
        subprocess.call(f'rm -f {header_path}', shell=True)
        subprocess.call(f'rm -f {cur_bitstream_path}', shell=True)
        return False
    
    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)

    return True
