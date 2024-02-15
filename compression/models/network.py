# @Author: Bin Duan <bduan2@hawk.iit.edu>

import tinycudann as tcnn
from models.quantizable import QuantizableModule
from utils.misc import get_param_partitions, generate_param_index_list
from tabulate import tabulate


class QuantizableNetworkWithInputEncoding(QuantizableModule):
    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config):
        super().__init__()
        self.net = tcnn.NetworkWithInputEncoding(n_input_dims, n_output_dims, encoding_config, network_config)
        params_per_level, params_per_layer = get_param_partitions(encoding_config, network_config, n_output_dims)
        self.param_per_level = params_per_level
        self.param_per_layer = params_per_layer
        self.param_partitions = generate_param_index_list(params_per_level + params_per_layer)
        
        # initialize quantization parameters
        self.init_quant_params()

        assert max(self.param_partitions) == sum(p.numel() for p in self.net.parameters() if p.requires_grad), \
            f"Parameter partitions {max(self.param_partitions)} do not match num of parameters in model {sum(p.numel() for p in self.net.parameters() if p.requires_grad)}"

        self.param_counts = params_per_level + params_per_layer # number of parameters for each level
        self.param_partitions = self.param_partitions[:-1]  # remove the last element, which is the total number of parameters


    def get_flops_str(self):
        flops_table = []
        for i in range(len(self.params_per_level)):
            flops_table.append([f"Enc level {i}", f"{self.params_per_level[i]}", ""])
        for i in range(len(self.params_per_layer)):
            flops_table.append([f"MLP Layer {i}", f"{self.params_per_layer[i]}", f"{2 * self.params_per_layer[i]}"])
        flops_table.append(["Quantization", f"{len(self.param_partitions)}", ""])

        self.flops_str = tabulate(flops_table, headers=["module", "#parameters or shape", "#flops"], tablefmt="github")

    def to_device(self, device):
        self = self.to(device)
        self.net.to(device)
        self._scale.to(device)
        self._mu.to(device)
        self._q_scale.to(device)