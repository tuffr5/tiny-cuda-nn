# @Author: Bin Duan <bduan2@hawk.iit.edu>

import copy
import math
import torch
from typing import Optional
from torch import nn, Tensor
from torch.distributions import Laplace

from utils.misc import MAX_AC_MAX_VAL, MIN_SCALE_NN_WEIGHTS_BIAS



class QuantizableModule(nn.Module):
    """This class is **not** made to be instantiated. It is thought as an interface
    from which all the modules should inherit. It implements all the mechanism to
    quantize, entropy code and measure the rate of the Module.
    """

    def __init__(self):
        """Instantiate a quantizable module with a list of available
        quantization steps.
        
        """
        super().__init__()
        # List of the available quantization steps (will be in the module class)
        self._POSSIBLE_Q_STEP = None

        # Store the full precision here by calling self.save_full_precision_param()
        self._full_precision_param: Optional[Tensor] = None

        # Store the quantization step info for the weight and biases
        self._q_step: Optional[int] = None

    def get_param(self) -> Tensor:
        """Return the weights and biases **currently** inside the Linear modules.
        """
        return self.params

    def set_param(self, param: Tensor):
        """Set the current parameters of the ARM.

        Args:
            param (OrderedDict[str, Tensor]): Parameters to be set.
        """
        self.params.copy_(param)

    def save_full_precision_param(self):
        """Store the full precision parameters inside an internal attribute
            self._full_precision_param.
        The current parameters **must** be the full precision parameters.
        """
        self._full_precision_param = copy.deepcopy(self.get_param())

    def get_full_precision_param(self) -> Tensor:
        """Return the **already saved** full precision parameters.
        They must have been saved with self.save_full_precision_param() beforehand!

        Returns:
            Tensor: The full precision parameters if available,
                None otherwise.
        """
        return self._full_precision_param

    def save_q_step(self, q_step: int):
        """Save a quantization step into an internal attribute self._q_step."""
        self._q_step = q_step

    def get_q_step(self) -> Optional[int]:
        """Return the quantization used to go from self._full_precision_param to the
        current parameters.

        Returns:
            The quantization step which has been used.
        """
        return self._q_step

    def measure_laplace_rate(self) -> float:
        """Get the rate associated with the current parameters.
        # ! No back propagation is possible in this method as we work with float,
        # ! not with tensor.
        """
        sent_param: []
        rate_param: 0.

        # We don't have a quantization step loaded which means that the parameters are
        # not yet quantized. Return zero rate.
        if self.get_q_step() is None:
            return rate_param

        # Quantization is round(parameter_value / q_step) * q_step
        sent_param = (self.get_param() / self.get_q_step())

        # compute their entropy.
        distrib = Laplace(0., max(sent_param.std().item() / math.sqrt(2), MIN_SCALE_NN_WEIGHTS_BIAS))
        # No value can cost more than 32 bits
        proba = torch.clamp(distrib.cdf(sent_param + 0.5) - distrib.cdf(sent_param - 0.5), min=2 ** -32, max=None)

        rate_param = -torch.log2(proba).sum().item()

        return rate_param

    def quantize(self, q_step) -> bool:
        """Quantize **in place** the model with a given quantization step q_step.
        The current model parameters are replaced by the quantized one.

        This methods save the q_step parameter as an attribute of the class

        Args:
            q_step: quantization step

        Return:
            bool: True if everything went well, False otherwise
        """
        fp_param = self.get_full_precision_param()

        assert fp_param is not None, 'You must save the full precision parameters '\
            'before quantizing the model. Use model.save_full_precision_param().'

        self.save_q_step(q_step)
        sent_param = torch.round(fp_param / self.get_q_step())
        if sent_param.abs().max() > MAX_AC_MAX_VAL:
            print(f'Sent param exceed MAX_AC_MAX_VAL! Q step {self.get_q_step()} too small.')
            return False

        q_param = sent_param * self.get_q_step()
        self.set_param(q_param)

        return True
