from aikido.__api__ import Aikidoka, Kata
from aikido.__api__.Dojo import DojoListener, DojoKun

try:
    from tqdm import tqdm
except ImportError:
    print("no 'tqdm' installation detected")


class TqdmListener(DojoListener):
    """DojoListener implementation which displays a tqdm progress bar."""

    def __init__(self, dan_progress:bool = True, batch_progress:bool = True):
        self.dan_progress = dan_progress
        self.batch_progress = batch_progress

    def training_started(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        if self.dan_progress:
            self.dan_bar = tqdm(total=kun.dans, desc="dans")
        if self.batch_progress:
            self.batch_bar = tqdm(total=len(kata), desc="batches")

    def training_finished(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        if self.dan_progress:
            self.dan_bar.close()
        if self.batch_progress:
            self.batch_bar.close()

    def dan_started(self, aikidoka: Aikidoka, run: (int, int)):
        if self.dan_progress:
            self.dan_bar.update(run[0])

    def batch_started(self, aikidoka: Aikidoka, batch, run: (int, int)):
        if self.batch_progress:
            self.batch_bar.update(run[0])
