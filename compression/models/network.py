# @Author: Bin Duan <bduan2@hawk.iit.edu>

import tinycudann as tcnn
from models.quantizable import QuantizableModule
from utils.misc import get_param_partitions, reverse_generate_param_index_list


class QuantizableNetworkWithInputEncoding(QuantizableModule):
    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config):
        super().__init__()
        self.net = tcnn.NetworkWithInputEncoding(n_input_dims, n_output_dims, encoding_config, network_config)
        self.param_partitions = get_param_partitions(encoding_config, network_config, n_output_dims)
        
        # initialize quantization parameters
        self.init_quant_params()

        assert max(self.param_partitions) == sum(p.numel() for p in self.net.parameters() if p.requires_grad), \
            f"Parameter partitions {max(self.param_partitions)} do not match num of parameters in model {sum(p.numel() for p in self.net.parameters() if p.requires_grad)}"
        print(f"Total num of parameters: Model({max(self.param_partitions)}) + Quant({2 * len(self.param_partitions)})")

        self.param_counts_per_level = reverse_generate_param_index_list(self.param_partitions) # number of parameters for each level
        self.param_partitions = self.param_partitions[:-1]  # remove the last element, which is the total number of parameters