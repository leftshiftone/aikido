from dataclasses import dataclass

import torch

from aikido.__api__ import Aikidoka, Kata
from aikido.__api__.Dojo import DojoListener, DojoKun


@dataclass
class OnnxListener(DojoListener):
    """
    DojoListener implementation which exports the aikido as an ONNX model.
    For more information see https://pytorch.org/docs/stable/onnx.html
    """

    model_name: str = "model.onnx"

    def inference_started(self, aikidoka: Aikidoka, x, batch_length: int):
        if self.dummy_input is None:
            self.dummy_input = torch.randn(x.shape, device='cuda')

    def training_finished(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        # Providing input and output names sets the display names for values
        # within the model's graph. Setting these does not change the semantics
        # of the graph; it is only for readability.
        #
        # The inputs to the network consist of the flat list of inputs (i.e.
        # the values you would pass to the forward() method) followed by the
        # flat list of parameters. You can partially specify names, i.e. provide
        # a list here shorter than the number of inputs to the model, and we will
        # only set that subset of names, starting from the beginning.
        input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
        output_names = ["output1"]

        torch.onnx.export(aikidoka, self.dummy_input, self.model_name, verbose=True, input_names=input_names,
                          output_names=output_names)
