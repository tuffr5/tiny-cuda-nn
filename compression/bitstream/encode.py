# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import os
import subprocess
import torch
import numpy as np

from bitstream.header import write_frame_header
from bitstream.range_coder import RangeCoder
from typing import Tuple

@torch.no_grad()
def encode_frame(model, bitstream_path: str, img_size: Tuple[int, int, int], config: dict):
    """Convert a model to a bitstream located at <bitstream_path>.

    Args:
        model: A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """

    torch.use_deterministic_algorithms(True)
    # After this: all params are quantized and are ready to be sent
    grid, mu, scale = model.prestep_for_entropy_encoding()
    model.eval()
    model.to('cpu')

    subprocess.call(f'rm -f {bitstream_path}', shell=True)

    # ================= Encode encoding and network params into a bitstream file ================ #
    acmax_scale_and_n_bytes = []
    for module_name in model.modules_to_send:
        acmax_i, scale_i = getattr(model, module_name).get_acmax_and_scale()
        range_coder = RangeCoder(AC_MAX_VAL=acmax_i)
        cur_bitstream_path = f'{bitstream_path}_{module_name}'
        if module_name == 'grid':
            scale_i = 0 # placeholder not used
            range_coder.encode(cur_bitstream_path, grid, mu, scale)
        else:
            weights = getattr(model, module_name).params
            range_coder.encode(
                cur_bitstream_path,
                weights,
                torch.zeros_like(weights),
                scale_i * torch.ones_like(weights)
            )
        acmax_scale_and_n_bytes.append((acmax_i, scale_i, os.path.getsize(cur_bitstream_path)))

    # Write the header
    header_path = f'{bitstream_path}_header'
    success = write_frame_header(
        header_path,
        img_size,
        config,
        acmax_scale_and_n_bytes
    )

    if success:
        # Concatenate everything inside a single file
        subprocess.call(f'rm -f {bitstream_path}', shell=True)
        subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {header_path}', shell=True)

        for module_name in model.modules_to_send:
            cur_bitstream_path = f'{bitstream_path}_{module_name}'
            subprocess.call(f'cat {cur_bitstream_path} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_bitstream_path}', shell=True)

    else:
        # remove intermediate bitstream
        subprocess.call(f'rm -f {header_path}', shell=True)

        for module_name in model.modules_to_send:
            cur_bitstream_path = f'{bitstream_path}_{module_name}'
            subprocess.call(f'cat {cur_bitstream_path} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_bitstream_path}', shell=True)

        return False

    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)

    return True
