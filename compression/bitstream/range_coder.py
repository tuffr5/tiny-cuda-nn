# Header INFO FOR COMPRESSED BITSTREAM
# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>


import torch
import constriction
import numpy as np
import torch

from torch import Tensor

from utils.misc import Q_PROBA_DEFAULT

class RangeCoder:
    def __init__(self, AC_MAX_VAL: int, Q_PROBA: int = Q_PROBA_DEFAULT):

        # Higher: more accurate but less reliable probability model
        # Actual q_step is 1 / Q_PROBA
        self.Q_PROBA = Q_PROBA

        # Data are in [-AC_MAX_VAL, AC_MAX_VAL - 1]
        self.AC_MAX_VAL = AC_MAX_VAL

        self.alphabet = np.arange(-self.AC_MAX_VAL, self.AC_MAX_VAL + 1)
        self.model_family = constriction.stream.model.QuantizedLaplace(-self.AC_MAX_VAL, self.AC_MAX_VAL + 1)

    def quantize_proba_parameters(self, x: Tensor) -> Tensor:
        """Apply a quantization to the input x to reduce floating point
        drift.

        Args:
            x (Tensor): The value to quantize

        Returns:
            Tensor: the quantize value
        """
        return torch.round(x * self.Q_PROBA) / self.Q_PROBA


    def encode(
        self,
        out_file: str,
        x: Tensor,
        mu: Tensor,
        scale: Tensor
    ):
        """Encode a 1D tensor x, using two 1D tensors mu and scale for the
        element-wise probability model of x.

        Args:
            x (Tensor): [B] tensor of values to be encoded
            mu (Tensor): [B] tensor describing the expectation of x
            scale (Tensor): [B] tensor with the standard deviations of x
        """

        mu = self.quantize_proba_parameters(mu)
        scale = self.quantize_proba_parameters(scale)

        x = x.numpy().astype(np.int32)
        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(x, self.model_family, mu, scale)

        with open(out_file, 'wb') as f_out:
            f_out.write(encoder.get_compressed())

    def load_bitstream_from_file(self, in_file: str):
        bitstream = np.fromfile(in_file, dtype=np.uint32)
        self.decoder = constriction.stream.queue.RangeDecoder(bitstream)

    def load_bitstream(self, bitstream: bytes):
        bitstream = np.frombuffer(bitstream, dtype=np.uint32)
        self.decoder = constriction.stream.queue.RangeDecoder(bitstream)

    def decode(self, mu: Tensor, scale: Tensor) -> Tensor:
        mu = self.quantize_proba_parameters(mu)
        scale = self.quantize_proba_parameters(scale)

        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)
        x = self.decoder.decode(self.model_family, mu, scale)

        x = torch.tensor(x).to(torch.float)

        return x