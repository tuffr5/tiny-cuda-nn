# @Author: Bin Duan <bduan2@hawk.iit.edu>

from torch import Tensor
import tinycudann as tcnn

from utils.misc import POSSIBLE_Q_STEP_NN
from models.quantizable import QuantizableModule


class EncodingModule(tcnn.Encoding, QuantizableModule):
    def __init__(self, n_input_dims, encoding_config, possible_q_steps: Tensor = POSSIBLE_Q_STEP_NN):
        """
        A module that performs encoding using the tcnn.Encoding class and supports quantization.

        Args:
            n_input_dims (int): Number of input dimensions.
            encoding_config (dict): Configuration for encoding.
            possible_q_steps (Tensor, optional): Possible quantization steps. Defaults to POSSIBLE_Q_STEP_NN.
        """
        super().__init__(n_input_dims, encoding_config)
        self._POSSIBLE_Q_STEP = possible_q_steps


class NetworkModule(tcnn.Network, QuantizableModule):
    def __init__(self, n_input_dims, n_output_dims, network_config, possible_q_steps: Tensor = POSSIBLE_Q_STEP_NN):
        """
        A module that represents a network using the tcnn.Network class and supports quantization.

        Args:
            n_input_dims (int): Number of input dimensions.
            n_output_dims (int): Number of output dimensions.
            network_config (dict): Configuration for the network.
            possible_q_steps (Tensor, optional): Possible quantization steps. Defaults to POSSIBLE_Q_STEP_NN.
        """
        super().__init__(n_input_dims, n_output_dims, network_config)
        self._POSSIBLE_Q_STEP = possible_q_steps


class NetwortwithInputEncoding(tcnn.NetworkWithInputEncoding, QuantizableModule):
    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, possible_q_steps: Tensor = POSSIBLE_Q_STEP_NN):
        """
        A module that represents a network with input encoding using the tcnn.NetworkWithInputEncoding class and supports quantization.

        Args:
            n_input_dims (int): Number of input dimensions.
            n_output_dims (int): Number of output dimensions.
            encoding_config (dict): Configuration for encoding.
            network_config (dict): Configuration for the network.
            possible_q_steps (Tensor, optional): Possible quantization steps. Defaults to POSSIBLE_Q_STEP_NN.
        """
        super().__init__(n_input_dims, n_output_dims, encoding_config, network_config)
        self._POSSIBLE_Q_STEP = possible_q_steps
