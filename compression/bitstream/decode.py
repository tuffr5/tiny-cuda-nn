# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>

from typing import Tuple
import torch

from torch import Tensor
from bitstream.header import read_frame_header
from bitstream.range_coder import RangeCoder
from models.network import NetwortwithInputEncoding
from utils.misc import generate_input_grid, POSSIBLE_SCALE_NN, POSSIBLE_Q_STEP_NN


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
    # To decode each NN (arm, upsampling and synthesis):
    #   1. Instantiate an empty Module
    #   2. Populate it with the weights and biases decoded from the bitstream
    #   3. Send it to the requested device.
                
    # initialize empty model
    model = NetwortwithInputEncoding(
        n_input_dims=2,
        n_output_dims=header_info.get('img_size')[-1],
        encoding_config=header_info.get('encoding_configs'),
        network_config=header_info.get('network_configs')
    )
    # =========================== Decode the NNs ============================ #
    
    range_coder_nn = RangeCoder(AC_MAX_VAL = header_info.get('ac_max_val_nn'))
    range_coder_nn.load_bitstream(bitstream)

    model_param_quant = range_coder_nn.decode(torch.zeros(model.params.shape), torch.ones(model.params.shape) * POSSIBLE_SCALE_NN[header_info.get('scale_index_nn')])
    
    # Don't forget inverse quantization!
    model_param = model_param_quant * POSSIBLE_Q_STEP_NN[header_info.get('q_step_index_nn')]
    model.set_param(model_param)

    xy = generate_input_grid(header_info.get('img_size'), device)

    return model(xy).reshape(header_info.get('img_size')).clamp(0.0, 1.0)