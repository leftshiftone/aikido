from abc import ABC, abstractmethod
from typing import List

from aikido.__common__.iterator.BatchIterator import BatchIterator


class Preprocessor:
    def preprocess(self, text: str):
        return text


class Kata(ABC):
    def __init__(self, df, batch_size: int, clipping: int, labels: List[str]):
        self.iterator = BatchIterator(df, batch_size, clipping)
        self.labels = labels

    def __len__(self):
        return self.iterator.__len__()

    def __iter__(self):
        return self.iterator.__iter__()

    def __next__(self):
        return self.iterator.__next__()

    @abstractmethod
    def split(self, ratio: float):
        pass

    def unapply(self):
        return self.iterator.df, self.iterator.batch_size, self.iterator.max_seq_len

    @property
    def label_names(self):
        return self.labels

    @property
    def label_count(self):
        return len(self.labels)

    def plot_barchart(self):
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#282828'

        plt.figure(figsize=(8, 6))
        self.iterator.df.groupby('label').label.count().plot.bar(ylim=0)
        plt.show()
