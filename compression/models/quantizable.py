# @Author: Bin Duan <bduan2@hawk.iit.edu>

import math
import torch
import torch.nn.functional as F
import tinycudann as tcnn
from torch.distributions import Laplace
from utils.misc import AC_MAX_VAL


def laplace_cdf(x, loc, scale):
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


def get_mu_scale(raw_proba_param):
    mu, log_scale = [tmp.view(-1) for tmp in torch.split(raw_proba_param, 1, dim=1)]
    print(f"mu: {mu.amin()}, {mu.amax()}")
    print(f"log_scale: {log_scale.amin()}, {log_scale.amax()}")
    scale = torch.exp(-0.5 * torch.clamp(log_scale, min=-10., max=13.8155))
    return mu, scale


def get_grid_context(grid, n_context):
    """pad the grid to the right size"""
    grid_context = torch.zeros((grid.shape[0], n_context), device=grid.device)
    grid_context[::2] =  F.pad(grid[::2], pad=(n_context, 0), mode='constant', value=-1.0).unfold(0, n_context, 1)[:-1]
    grid_context[1::2] = F.pad(grid[1::2], pad=(n_context, 0), mode='constant', value=1.0).unfold(0, n_context, 1)[:-1]
    
    return grid_context


class QModule:
    def __init__(self):
        self.fpfm = None

    def measure_laplace_rate(self):
        raise NotImplementedError("Subclass must implement this method")
    
    @torch.no_grad()
    def quantize(self):
        for p in self.parameters():
            p.data = (p.data / self.fpfm).round() * self.fpfm
    
    @torch.no_grad()
    def prestep_for_entropy_encoding(self):
        for p in self.parameters():
            p.data = (p.data / self.fpfm).round().clamp(-AC_MAX_VAL, AC_MAX_VAL+1)

    @torch.no_grad()
    def poststep_for_entropy_decoding(self):
        for p in (self.parameters()):
            p.data = p * self.fpfm

    def get_acmax_and_scale(self):
        return (int(self.params.abs().max().item()), self.params.std().item() / math.sqrt(2))
    
    def set_params(self, params):
        self.params.data = params
        

class QGrid(QModule, tcnn.Encoding):
    def __init__(self, n_input_dims, config):
        QModule.__init__(self)
        tcnn.Encoding.__init__(self, n_input_dims, config)
        self.config = config
        self.fpfm = 1.0 / 16.0

    def measure_laplace_rate(self, x, raw_proba_param):
        mu, scale = get_mu_scale(raw_proba_param)
        print(f"estimated mu: {mu}, estimated scale: {scale}")
        proba = torch.clamp_min(
            laplace_cdf(x + 0.5, mu, scale) - laplace_cdf(x - 0.5, mu, scale),
            min=2 ** -16
            )
        rate = -torch.log2(proba).sum()
        return rate


class QNetwork(QModule, tcnn.Network):
    def __init__(self, n_input_dims, n_output_dims, config):
        QModule.__init__(self)
        tcnn.Network.__init__(self, n_input_dims, n_output_dims, config)
        self.config = config
        self.fpfm = 1.0 / 128.0
    
    def measure_laplace_rate(self):
        param = self.params.clone()
        param = param / self.fpfm # actually doing quantization here
        distrib = Laplace(0., max(param.std() / math.sqrt(2), 1.0 / 1024.0))
        proba = torch.clamp(
            distrib.cdf(param + 0.5) - distrib.cdf(param - 0.5), 
            min=2 ** -32, 
            max=None
            )
        rate = -torch.log2(proba).sum()
            
        return rate