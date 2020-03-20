import logging
from dataclasses import dataclass

import torch

from aikido.__api__ import Aikidoka, Kata
from aikido.__api__.Dojo import DojoListener, DojoKun


@dataclass
class BackendListener(DojoListener):
    """DojoListener implementation which configures torch backends."""

    backend:str
    cudnn_autotune:bool = False

    def training_started(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        def setup_cuda():
            if 'cudnn' in self.backend:
                torch.backends.cudnn.enabled = True
                if self.cudnn_autotune:
                    torch.backends.cudnn.benchmark = True
            else:
                torch.backends.cudnn.enabled = False

        def setup_cpu():
            if 'mkl' in self.backend and 'mkldnn' not in self.backend:
                torch.backends.mkl.enabled = True
            elif 'mkldnn' in self.backend:
                raise ValueError("MKL-DNN is not supported yet.")
            elif 'openmp' in self.backend:
                torch.backends.openmp.enabled = True

        logging.info("initialize backend '" + self.backend + "'")

        setup_cuda()
        setup_cpu()
