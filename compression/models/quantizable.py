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
        self.net_shadow = None

        # full precision parameters used in model (-1, 1)
        self._full_precision_param: Optional[list] = None
        # quantized parameters used in entropy coding (0, AC_MAX_VAL)
        self._quantized_precision_param: Optional[list] = None
        
        # quantization parameters for each level
        self._mu: Optional[nn.Parameter] = None
        self._scale: Optional[nn.Parameter] = None
        
        # used in entropy coding
        self._q_scale: Optional[nn.Parameter] = None

        self.param_partitions: Optional[list] = None

    def init_quant_params(self, fp_params: Optional[list] = None):
        assert self.param_partitions is not None, 'You must initialize the parameter partitions before initializing the quantization parameters.'
        if fp_params is not None:
            fp_params = fp_params.detach().clone()
            params = self.fragment_param(fp_params)
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
        return self.net.params.clone()

    def set_param(self, param: Tensor):
        self.net.params = nn.Parameter(param)
    
    def fragment_param(self, param: Tensor) -> list[Tensor]:
        return torch.tensor_split(param, self.param_partitions)

    def save_full_precision_param(self):
        """Save the detach full precision parameters."""
        self._full_precision_param = self.get_param()

    def get_full_precision_param(self) -> Tensor:
        return self._full_precision_param
    
    def save_quantized_precision_param(self, param: Tensor):
        self._quantized_precision_param = param

    def get_quantized_precision_param(self) -> Tensor:    
        return self._quantized_precision_param
    
    def sync_shadow(self):
        self.net_shadow = copy.deepcopy(self.net)

    def measure_laplace_rate(self) -> float:     
        rate = 0.

        if self._mu is None or self._scale is None:
            return rate
        
        param = self.fragment_param(self.get_param())  
        for p, mu, scale in zip(param, self._mu, self._scale):
            # No value can cost more than 16 bits
            proba = torch.clamp_min(laplace_cdf(p + 0.5, mu*scale, scale) - laplace_cdf(p - 0.5, mu*scale, scale), min=2 ** -16)

            rate += -torch.log2(proba).sum()
            
        return rate
    
    def forward(self, x: Tensor, quant: bool= False) -> Tensor:
        if quant:
            self.sync_shadow()
            cur_param = self.fragment_param(self.get_param())

            rate = 0.
            params = []
            scale_grad = []
            for p, mu, scale in zip(cur_param, self._mu, self._scale):
                # autograd for rate loss
                proba = torch.clamp_min(laplace_cdf(p + 0.5, mu*scale, scale) - laplace_cdf(p - 0.5, mu*scale, scale), min=2 ** -16)
                rate += -torch.log2(proba).sum()

                # we will calulate the gradient of scale manually
                p = p.detach().clone()
                scale = scale.detach().clone()
                quant_param = (round_ste(p / scale) + mu).clamp(0, AC_MAX_VAL)
                params.append((quant_param - mu) * scale)
                scale_grad.append((round_ste(p / scale) - p / scale).mean())
            
            # hack: autograd the modified parameters regards to l2 loss
            self.net_shadow.params = nn.Parameter(torch.cat(params).flatten())
            
            return self.net_shadow(x), rate, torch.tensor(scale_grad, device=x.device).unsqueeze(1)
        
        return self.net(x)
    
    @torch.no_grad()
    def quantize(self):
        """Quantize the model. Should be called after all training steps."""

        # this param is actually quantized and dequantized
        param = self.get_param()

        param = self.fragment_param(param)
        params = []
        for i, p, mu, scale in zip(range(len(self._mu)), param, self._mu, self._scale):
            quant_param = ((p / scale).round() + mu).clamp(0, AC_MAX_VAL)

            # save q_scale for entropy coding
            self._q_scale[i] = p.std() / scale

            params.append(quant_param)
        
        self.save_quantized_precision_param(torch.cat(params).flatten())

    @torch.no_grad()
    def dequantize(self):
        """Dequantize the model. Make sure run quantize() before dequantize()."""
        sent_param = self.get_quantized_precision_param()
        assert sent_param is not None, 'You must quantize the model before dequantizing it. '\
            'Use model.quantize().'
        sent_param = self.fragment_param(sent_param)
        params = []
        for quant_param, mu, scale in zip(sent_param, self._mu, self._scale):
            params.append((quant_param - mu) * scale)

        self.set_param(torch.cat(params).flatten())