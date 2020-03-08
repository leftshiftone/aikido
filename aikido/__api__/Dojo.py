from abc import ABC

from aikido.__api__.Aikidoka import Aikidoka
from aikido.__api__.Kata import Kata


class DojoListener:
    """
    Instances which extends from this class can be used to listen to certain dojo events.
    """

    def dan_started(self, aikidoka: Aikidoka, run: (int, int)):
        pass

    def dan_finished(self, aikidoka: Aikidoka, run: (int, int), metrics: (float, float)):
        pass

    def training_started(self, aikidoka: Aikidoka, kata: Kata, kun: 'DojoKun'):
        pass

    def training_finished(self, aikidoka: Aikidoka, kata: Kata, kun: 'DojoKun'):
        pass

    def batch_started(self, aikidoka: Aikidoka, batch, run: (int, int)):
        pass

    def batch_finished(self, aikidoka: Aikidoka, batch, run: (int, int)):
        pass


class Evaluation:

    def __init__(self, labels, values, rowids, isprop: True):
        self.labels = labels
        self.values = values
        self.rowids = rowids
        self.isprop = isprop

    def __iter__(self):
        for i in [self.labels, self.values, self.rowids, self.isprop]:
            yield i


class DojoKun:
    def __init__(self, optimizer, loss, dans=20):
        self.optimizer = optimizer
        self.loss = loss
        self.dans = dans


class Dojo(ABC):

    def add_listener(self, listener: DojoListener):
        pass

    def train(self, aikidoka: Aikidoka, kata: Kata):
        """
        Trains the given aikidoka with the given kata.
        If you want to get more detailed information about the training progress
        register a proper listener instance.
        """
        pass

    def evaluate(self, aikidoka: Aikidoka, kata: Kata, probability: bool = True) -> Evaluation:
        """
        Evaluates the given aikidoka with the given kata. Returns a tuple containing the expected and predicted labels.
        Merges predictions with the same identifier column if the "merge" attribute is set to True.
        """
        pass
