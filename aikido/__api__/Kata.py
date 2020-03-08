from abc import ABC, abstractmethod

from aikido.__common__.iterator.BatchIterator import BatchIterator


class Preprocessor:
    def preprocess(self, text:str):
        return text


class Kata(ABC):
    def __init__(self, df, batch_size:int, clipping:int):
        self.iterator = BatchIterator(df, batch_size, clipping)

    def __len__(self):
        return self.iterator.__len__()

    def __iter__(self):
        return self.iterator.__iter__()

    def __next__(self):
        return self.iterator.__next__()

    @abstractmethod
    def split(self, ratio:float):
        pass

    def unapply(self):
        return self.iterator.df, self.iterator.batch_size, self.iterator.max_seq_len
