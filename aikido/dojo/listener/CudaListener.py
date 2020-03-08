import logging
from torch import cuda

from aikido.__api__ import Aikidoka, Kata
from aikido.__api__.Dojo import DojoListener, DojoKun


class CudaListener(DojoListener):
    """DojoListener implementation which enables CUDA if available."""

    def training_started(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        if cuda.is_available():
            logging.info("enable CUDA for aikidoka")
            aikidoka.cuda()
        else:
            logging.info("enable CPU for aikidoka")
