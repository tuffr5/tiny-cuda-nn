# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import math
import torch
from typing import Tuple

from torch import Tensor
from bitstream.header import read_frame_header
from bitstream.range_coder import RangeCoder
from models.network import QuantizableNetworkWithInputEncoding
from utils.misc import generate_input_grid, MAX_AC_MAX_VAL


@torch.no_grad()
def decode_frame(
    bitstream_path: str,
    device: str
) -> Tuple[Tensor, bytes, int]:
    """Decode a bitstream located at <bitstream_path>.
    This only performs the cool-chic part of the decoding. Inter coding module goes after
    that in decode_video. 

    Args:
        bitstream_path (str): Absolute path of the bitstream. We keep that to output some temporary file
            at the same location than the bitstream itself

    Returns:
        Tensor: The decoded image in [0., 1.]
    """
    torch.use_deterministic_algorithms(True)

    with open(bitstream_path, 'rb') as fin:
        bitstream = fin.read()

    # ========================== Parse the header =========================== #
    header_info = read_frame_header(bitstream)
    # ========================== Parse the header =========================== #

    # Read all the bytes and discard the header
    bitstream = bitstream[header_info.get('n_bytes_header'): ]

    # =========================== Decode the NNs ============================ #
    #   1. Instantiate an empty Module
    #   2. Populate it with the weights decoded from the bitstream
    #   3. Send it to the requested device.
                
    # initialize empty model
    model = QuantizableNetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=header_info.get('img_size')[-1],
        encoding_config=header_info.get('encoding_configs'),
        network_config=header_info.get('network_configs')
    )
    # =========================== Decode the encoding and network ============ #
    params = []
    range_coder = RangeCoder(AC_MAX_VAL = MAX_AC_MAX_VAL)
    for shape_i, mu_i, scale_i, n_bytes_i in header_info.get("shape_mu_scale_and_n_bytes"):
        range_coder.load_bitstream(bitstream[:n_bytes_i])
        model_param_quant_i = range_coder.decode(torch.zeros(shape_i), torch.ones(shape_i))
        model_param_quant_i = (model_param_quant_i - mu_i) * scale_i 
        params.append(model_param_quant_i)
        bitstream = bitstream[n_bytes_i:]

    model.set_param(torch.cat(params).flatten())

    xy = generate_input_grid(header_info.get('img_size'), device)

    return model(xy).reshape(header_info.get('img_size')).clamp(0.0, 1.0)