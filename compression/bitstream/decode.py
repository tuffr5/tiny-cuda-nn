# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import torch
from typing import Tuple

from torch import Tensor
from bitstream.header import read_frame_header
from bitstream.range_coder import RangeCoder
from models.network import QuantizableNetworkWithInputEncoding
from models.quantizable import get_mu_scale
from utils.misc import generate_input_grid
import numpy as np


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
    arm_config = header_info.get('arm_configs')
    arm_config['n_contexts'] = header_info.get('n_contexts')
    model = QuantizableNetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=header_info.get('img_size')[-1],
        encoding_config=header_info.get('encoding_configs'),
        network_config=header_info.get('network_configs'),
        arm_config=arm_config
    )
    n_contexts = header_info.get('n_contexts')
    for module_name, (acmax_i, scale_i, n_bytes_i) in zip(model.modules_to_send, header_info.get("acmax_scale_and_n_bytes")):
        range_coder = RangeCoder(AC_MAX_VAL = acmax_i)
        range_coder.load_bitstream(bitstream[:n_bytes_i])
        weights = getattr(model, module_name).params.data
        shape_ = weights.shape[0]
        if module_name == 'grid':
            weights = torch.zeros(shape_ + n_contexts)
            for i in range(shape_):
                cur_context = weights[i:i+n_contexts]
                cur_context = cur_context.unsqueeze_(0).to(device)
                raw_proba_param = model.arm(cur_context)
                cur_mu, cur_scale = get_mu_scale(raw_proba_param)
                cur_grid = range_coder.decode(cur_mu.cpu(), cur_scale.cpu())
                weights[n_contexts+i] = cur_grid
                
            weights = weights[n_contexts:]
            weights = weights.to(device)
        else:
            weights = range_coder.decode(torch.zeros_like(weights.cpu()), torch.ones_like(weights.cpu()) * scale_i).to(device) 

        getattr(model, module_name).set_params(weights)
        getattr(model, module_name).poststep_for_entropy_decoding()
        bitstream = bitstream[n_bytes_i:]

    xy = generate_input_grid(header_info.get('img_size'), device)

    return model(xy).reshape(header_info.get('img_size')).clamp(0.0, 1.0)