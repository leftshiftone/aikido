import logging
from dataclasses import dataclass

import torch

from aikido.__api__ import Aikidoka
from aikido.__api__.Dojo import DojoListener


@dataclass
class CudaMemoryListener(DojoListener):
    """DojoListener implementation which monitors the CUDA memory."""

    optimize_on_dan: bool = False
    optimize_on_batch: bool = False
    optimize_on_inference: bool = False

    def dan_started(self, aikidoka: Aikidoka, run: (int, int)):
        logging.debug("")
        logging.debug("cuda memory (dan started)")
        for i in range(torch.cuda.device_count()):
            logging.debug('cuda:' + str(i) + ' (allocated):', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
            logging.debug('cuda:' + str(i) + ' (cached):   ', round(torch.cuda.memory_cached(i)/1024**3,1), 'GB')

    def dan_finished(self, aikidoka: Aikidoka, run: (int, int), metrics: (float, float)):
        if self.optimize_on_dan:
            torch.cuda.empty_cache()

    def batch_finished(self, aikidoka: Aikidoka, batch, run: (int, int)):
        if self.optimize_on_batch:
            torch.cuda.empty_cache()

    def inference_finished(self, aikidoka: Aikidoka, x, batch_length: int, y):
        if self.optimize_on_inference:
            torch.cuda.empty_cache()
