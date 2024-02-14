# @Author: Bin Duan <bduan2@hawk.iit.edu>

import copy
import math
import torch
from typing import Optional
from torch import nn, Tensor
from torch.distributions import Laplace
from utils.misc import MAX_AC_MAX_VAL


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class QuantizableModule(nn.Module):
    """Base class for quantizable modules. It provides the basic functionalities."""
    def __init__(self):
        super().__init__()

        self.net = None
        self._full_precision_param: Optional[list] = None
        self._quantized_precision_param: Optional[list] = None
        
        # quantization parameters for each level
        self._mu = None
        self._scale: Optional[nn.ParameterList] = None

        self.param_partitions: Optional[list] = None

    def init_quant_params(self, params: Optional[list] = None):
        assert self.param_partitions is not None, 'You must initialize the parameter partitions before initializing the quantization parameters.'
        if params is not None:
            params = self.fragment_param(params)
            p_max = torch.tensor([p.abs().max() for p in params])
            p_min = torch.tensor([-p.abs().max() if p.min() < 0 else 0 for p in params])
            self._scale = nn.Parameter((p_max - p_min) / (MAX_AC_MAX_VAL + 1), requires_grad=True)
            self._mu = -p_min / torch.max(self._scale.detach(), torch.tensor(1e-5))
            self._mu = self._mu.round_()
        else:
            self._mu = torch.zeros(len(self.param_partitions))
            self._scale = nn.Parameter(torch.rand(len(self.param_partitions), 1), requires_grad=True)

    def get_param(self) -> Tensor:
        return self.net.params

    def set_param(self, param: Tensor):
        self.net.params.copy_(param)
    
    def fragment_param(self, param: Tensor) -> list[Tensor]:
        return torch.tensor_split(param, self.param_partitions)

    def save_full_precision_param(self):
        self._full_precision_param = copy.deepcopy(self.get_param())

    def get_full_precision_param(self) -> Tensor:
        return self._full_precision_param
    
    def save_quantized_precision_param(self, param: Tensor):
        self._quantized_precision_param = param

    def get_quantized_precision_param(self) -> Tensor:    
        return self._quantized_precision_param

    def measure_laplace_rate(self) -> float:     
        rate_param = 0.

        if self._mu is None or self._scale is None:
            return rate_param
        
        param = self.fragment_param(self.get_param())  
        for p, mu, scale in zip(param, self._mu, self._scale):
            sent_param = (p - mu) / scale

            # compute their entropy.
            distrib = Laplace(0., max(sent_param.std() / math.sqrt(2), 1.0e-3))
            # No value can cost more than 32 bits
            proba = torch.clamp(distrib.cdf(sent_param + 0.5) - distrib.cdf(sent_param - 0.5), min=2 ** -32, max=None)

            rate_param += -torch.log2(proba).sum()
            
        return rate_param

    def quantize(self):
        fp_param = self.get_full_precision_param()

        assert fp_param is not None, 'You must save the full precision parameters '\
            'before quantizing the model. Use model.save_full_precision_param().'
        
        fp_param = self.fragment_param(fp_param.detach())
        params = []
        for p, mu, scale in zip(fp_param, self._mu, self._scale):
            # print(f"fp: {torch.mean(p)}, {torch.std(p)}, {torch.amax(p)}, {torch.amin(p)}")
            # print(f"[Before]mu: {torch.amax(mu)}, scale: {torch.amax(scale)}")
            # print(f"[Before]mu: {torch.amin(mu)}, scale: {torch.amin(scale)}")
            # scale = scale * 1.0 / math.sqrt(p.numel() * MAX_AC_MAX_VAL)
            sent_param = (round_ste(p / scale) + mu).clamp(-MAX_AC_MAX_VAL, MAX_AC_MAX_VAL + 1)
            sent_param = (sent_param - mu) * scale

            # print(f"[After]mu: {torch.amax(mu)}, scale: {torch.amax(scale)}, sent_param: {torch.amax(sent_param)}")
            # print(f"[After]mu: {torch.amin(mu)}, scale: {torch.amin(scale)}, sent_param: {torch.amin(sent_param)}")

            params.append(sent_param)
        
        self.save_quantized_precision_param(torch.cat(params).flatten())
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)