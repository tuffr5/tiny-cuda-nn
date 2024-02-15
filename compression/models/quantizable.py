# @Author: Bin Duan <bduan2@hawk.iit.edu>

import copy
import math
import torch
from typing import Optional
from torch import nn, Tensor
from utils.misc import AC_MAX_VAL


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def laplace_cdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        loc (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


class QuantizableModule(nn.Module):
    """Base class for quantizable modules. It provides the basic functionalities."""
    def __init__(self):
        super().__init__()

        self.net = None
        self._full_precision_param: Optional[list] = None
        self._quantized_precision_param: Optional[list] = None
        
        # quantization parameters for each level
        self._mu: Optional[Tensor] = None
        self._scale: Optional[nn.ParameterList] = None
        
        # used in entropy coding
        self._q_scale: Optional[Tensor] = None

        self.param_partitions: Optional[list] = None

    def init_quant_params(self, params: Optional[list] = None):
        assert self.param_partitions is not None, 'You must initialize the parameter partitions before initializing the quantization parameters.'
        if params is not None:
            params = self.fragment_param(params.detach())
            for i, p in enumerate(params):
                if p.min() < 0:
                    p_min = -p.abs().max()
                else:
                    p_min = 0
                self._scale[i] = (p.abs().max() - p_min) / AC_MAX_VAL
            self._mu[:] = (-p_min / self._scale).round_()
        else:
            self._scale = nn.Parameter(torch.rand(len(self.param_partitions), 1), requires_grad=True)
            self._mu = nn.Parameter(torch.rand(len(self.param_partitions), 1), requires_grad=False)

        self._q_scale = nn.Parameter(torch.rand(len(self._mu), 1), requires_grad=False)

    def get_param(self) -> Tensor:
        return self.net.params

    def set_param(self, param: Tensor):
        self.net.params[:] = param
    
    def fragment_param(self, param: Tensor) -> list[Tensor]:
        return torch.tensor_split(param, self.param_partitions)

    def save_full_precision_param(self):
        """Save the detach full precision parameters."""
        self._full_precision_param = copy.deepcopy(self.get_param().detach())

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
            # No value can cost more than 16 bits
            proba = torch.clamp_min(laplace_cdf(p + 0.5, mu*scale, scale) - laplace_cdf(p - 0.5, mu*scale, scale), min=2 ** -16)

            rate_param += -torch.log2(proba).sum()
            
        return rate_param
    
    def forward(self, x: Tensor, quant: bool= False) -> Tensor:
        if quant:
            fp_param = self.fragment_param(self.get_full_precision_param())
            params = []
            for p, mu, scale in zip(fp_param, self._mu, self._scale):
                quant_param = (round_ste(p / scale) + mu).clamp(0, AC_MAX_VAL)
                params.append((quant_param - mu) * scale)
            
            self.set_param(torch.cat(params).flatten())
        
        return self.net(x)
    
    @torch.no_grad()
    def quantize(self):
        """Quantize the model. Should be called after all training steps."""
        fp_param = self.get_full_precision_param()

        assert fp_param is not None, 'You must save the full precision parameters '\
            'before quantizing the model. Use model.save_full_precision_param().'
        
        fp_param = self.fragment_param(fp_param)
        params = []
        for i, p, mu, scale in zip(range(len(self._mu)), fp_param, self._mu, self._scale):
            quant_param = ((p / scale).round() + mu).clamp(0, AC_MAX_VAL)

            # save q_scale for entropy coding
            self._q_scale[i] = p.std() / scale

            params.append(quant_param)
        
        self.save_quantized_precision_param(torch.cat(params).flatten())

    @torch.no_grad()
    def dequantize(self):
        """Dequantize the model."""
        sent_param = self.get_quantized_precision_param()
        assert quant_param is not None, 'You must quantize the model before dequantizing it. '\
            'Use model.quantize().'
        sent_param = self.fragment_param(sent_param)
        params = []
        for quant_param, mu, scale in zip(sent_param, self._mu, self._scale):
            params.append((quant_param - mu) * scale)

        self.set_param(torch.cat(params).flatten())