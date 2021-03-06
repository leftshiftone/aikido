import logging
from dataclasses import dataclass

import torch

from aikido.__api__ import Aikidoka, Kata
from aikido.__api__.Dojo import DojoListener, DojoKun


@dataclass
class SeedListener(DojoListener):
    """DojoListener implementation which initializes all random value generators with a seed."""
    seed: int

    def initialize(self, aikidoka: Aikidoka, kata: Kata, kun: 'DojoKun'):
        if self.seed >= 0:
            logging.debug("set seed to " + str(self.seed))

            from random import seed
            seed(self.seed)

            torch.manual_seed(self.seed)
            # noinspection PyUnresolvedReferences
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            try:
                import numpy as np
                np.random.seed(self.seed)
            except ImportError:
                pass
