import numpy as np
import pandas as pd
from sklearn.utils import resample

from aikido.__api__.Kata import Kata


class DirectKata(Kata):
    """
    Kata implementation which loads the data from an csv file.
    """

    def __init__(self, df, batch_size: int, clipping: int):
        super().__init__(df, batch_size, clipping)

    """
    Load a kata instance by using the given dataframe. The kata data can be upsampled by setting the 'upsample'
    attribute to true.
    """
    @staticmethod
    def from_dataframe(df, upsample:bool, seed:int, batch_size:int = 64, clipping:int = 100):
        labels = df[df.labels]
        df = df.sample(frac=1) if upsample is not True else DirectKata._upsample(df, labels, seed)
        return DirectKata(df, batch_size, clipping)


    @staticmethod
    def _upsample(df, labels, seed: int):
        max_size = 0

        for i in range(1, len(labels)):
            max_size = max(max_size, len(df[df.label == i]))

        array = []
        for i in range(1, len(labels)):
            df_minority = df[df.label == i]
            df_minority_upsampled = resample(df_minority, replace=True, n_samples=max_size, random_state=seed)
            array.append(df_minority_upsampled)

        df = pd.concat(array)
        return df.sample(frac=1)

    @staticmethod
    def _pad(rows, length:int, padding_idx:int = 100000):
        while len(rows) < length:
            rows.append(padding_idx)
        return list

    def split(self, ratio: float):
        """
        Splits this kata into two separate kata instances considering the split ratio.
        """
        (df, batch_size, clipping) = self.unapply()
        df1, df2 = np.split(df.sample(frac=1), [int(ratio * len(df))])
        return (DirectKata(df1, batch_size, clipping), DirectKata(df2, batch_size, clipping))
