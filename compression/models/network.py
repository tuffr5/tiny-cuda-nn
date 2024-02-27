# @Author: Bin Duan <bduan2@hawk.iit.edu>

import torch
import math
import copy
from models.quantizable import QNetwork, QGrid, get_grid_context, get_mu_scale
from models.quantizer import NoiseQuantizer, STEQuantizer


class QuantizableNetworkWithInputEncoding(torch.nn.Module):
    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, arm_config):
        super().__init__()
        self.grid = QGrid(n_input_dims, encoding_config)
        self.grid_shadow = torch.nn.Parameter(torch.zeros_like(self.grid.params))
        self.net = QNetwork(self.grid.n_output_dims, n_output_dims, network_config)
        self.arm = QNetwork(arm_config["n_contexts"], 2, arm_config) # arm_config["n_contexts"] is the number of context points
        self.noise_quantizer = NoiseQuantizer()
        self.ste_quantizer = STEQuantizer()

        self.modules_to_send = ['arm', 'net', 'grid']

    def _sync_grid(self):
        self.grid_shadow.data = copy.deepcopy(self.grid.params.data)

    def quantize_model(self):
        self.net.quantize()
        self.arm.quantize()

    def prestep_for_entropy_encoding(self):
        # return grid, mu, scale before entropy encoding
        grid = self.grid.params.detach().clone()
        grid = torch.round(grid / self.grid.fpfm)
        print(f"grid: {grid.amin()}, {grid.amax()}")
        grid_context = get_grid_context(grid, self.arm.config["n_contexts"])
        raw_proba_param = self.arm(grid_context).to(grid_context.dtype)
        mu, scale = get_mu_scale(raw_proba_param)

        for module_name in self.modules_to_send:
            getattr(self, module_name).prestep_for_entropy_encoding()

        return grid.cpu(), mu.cpu(), scale.cpu()

    def forward(self, x, training=False, STE=False):
        if training:
            self._sync_grid()
            if STE:
                grid = self.ste_quantizer(self.grid_shadow / self.grid.fpfm)
            else:
                grid = self.noise_quantizer(self.grid_shadow / self.grid.fpfm)

            grid_context = get_grid_context(grid, self.arm.config["n_contexts"])
            raw_proba_param = self.arm(grid_context)
            
            rate = 0.
            for module_name in self.modules_to_send:
                if module_name == 'grid':
                    rate += getattr(self, module_name).measure_laplace_rate(grid, raw_proba_param)
                else:
                    rate += getattr(self, module_name).measure_laplace_rate()

            return self.net(self.grid(x)), rate
        else:
            return self.net(self.grid(x))