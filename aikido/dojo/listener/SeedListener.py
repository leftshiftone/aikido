from dataclasses import dataclass

import torch

from aikido.__api__ import Aikidoka, Kata
from aikido.__api__.Dojo import DojoListener, DojoKun


@dataclass
class SeedListener(DojoListener):
    """DojoListener implementation which initializes all random value generators with a seed."""
    seed: int

    def training_started(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
