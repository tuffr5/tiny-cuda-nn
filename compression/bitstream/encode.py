# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import os
import subprocess
import torch
import math

from bitstream.header import write_frame_header
from bitstream.range_coder import RangeCoder
from typing import Tuple
from utils.misc import (
    get_num_layers, 
    get_num_levels, 
    MAX_HASH_LEVEL_BYTES,
    MAX_NN_BYTES, 
    AC_MAX_VAL, 
    MAX_PARAM_SHAPE
    )


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

    # ================= Encode encoding and network params into a bitstream file ================ #
    shape_mu_scale_and_n_bytes = []
    range_coder = RangeCoder(AC_MAX_VAL=AC_MAX_VAL)
    for i, pp_i, param_quant_i in zip(range(get_num_levels(config["encoding"]) + get_num_layers(config["network"])), 
                                      model.param_counts, 
                                      model.fragment_param(model.get_quantized_precision_param().cpu())):
        
        if pp_i > MAX_PARAM_SHAPE:
            raise ValueError(f"Found param shape {pp_i} exceeds the maximum allowed {MAX_PARAM_SHAPE}")
        
        mu_i = model._mu[i]
        scale_i = model._scale[i]
        q_scale_i = model._q_scale[i]
        cur_bitstream_path = f'{bitstream_path}_{i}'
        range_coder.encode(
            cur_bitstream_path,
            param_quant_i,
            mu_i * torch.ones(pp_i),
            q_scale_i * torch.ones(pp_i)
        )

        n_bytes_i = os.path.getsize(cur_bitstream_path)
        limitation = MAX_HASH_LEVEL_BYTES if i < get_num_levels(config["encoding"]) else MAX_NN_BYTES
        if n_bytes_i > limitation:
            raise ValueError(f"Found number of bytes {n_bytes_i} exceeds the maximum allowed {limitation}")
        
        # hack to encode 65536 to 0
        pp_i = 0 if pp_i == MAX_PARAM_SHAPE else pp_i
        n_bytes_i = 0 if n_bytes_i == limitation else n_bytes_i
        shape_mu_scale_and_n_bytes.append((pp_i, mu_i.item(), scale_i.item(), q_scale_i.item(), n_bytes_i))

    # Write the header
    header_path = f'{bitstream_path}_header'
    success = write_frame_header(
        header_path,
        img_size,
        config,
        shape_mu_scale_and_n_bytes
    )

    if success:
        # Concatenate everything inside a single file
        subprocess.call(f'rm -f {bitstream_path}', shell=True)
        subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {header_path}', shell=True)

        for i in range(get_num_levels(config["encoding"]) + get_num_layers(config["network"])):
            cur_bitstream_path = f'{bitstream_path}_{i}'
            subprocess.call(f'cat {cur_bitstream_path} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_bitstream_path}', shell=True)

    else:
        # remove intermediate bitstream
        subprocess.call(f'rm -f {header_path}', shell=True)

        for i in range(get_num_levels(config["encoding"]) + get_num_layers(config["network"])):
            cur_bitstream_path = f'{bitstream_path}_{i}'
            subprocess.call(f'cat {cur_bitstream_path} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_bitstream_path}', shell=True)

        return False

    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)

    return True
