import torch
import torch.nn.functional as F
import warnings
from torch.nn import Parameter, Module
from typing import Collection


class WeightDropout(torch.nn.Module):
    """A module that wraps another layer in which some weights will be replaced by 0 during training"""


    def __init__(self, module:Module, prob:float=0, layer_names:Collection[str] = ['weight_hh_l0']):
        super().__init__()
        self.module = module
        self.layer_names = layer_names
        self.prob = prob
        for layer in self.layer_names:
            # copy the weights of the selected layer
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', Parameter(w.data, False))
            self.module._parameters[layer] = F.dropout(w, p=self.prob, training=False)

    def _setweights(self):
        """Apply dropout to the raw weights"""
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.prob, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # avoid the warning due to weights are't flattened
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.prob, training=False)
        if hasattr(self.module, 'reset'):self.module.reset()
