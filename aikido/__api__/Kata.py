from abc import ABC
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame

from aikido.__api__.annotation import Experimental


class Preprocessor:
    def preprocess(self, text: str):
        return text


class Kata(ABC):
    def __init__(self, df: DataFrame, labels: List[str]):
        self.df = df.sample(frac=1)
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def split(self, ratio: float):
        """
        Splits this kata into two separate katas considering the split ratio.
        """
        df1, df2 = np.split(self.df.sample(frac=1), [int(ratio * len(self.df))])
        return Kata(df1, self.labels), Kata(df2, self.labels)

    @property
    def label_names(self):
        return self.labels

    @property
    def label_count(self):
        return len(self.labels)

    def upsample(self, seed: int = 123):
        # FIXME: scikitlearn dependency
        from sklearn.utils import resample

        max_size = 0

        for i in range(1, len(self.labels) + 1):
            max_size = max(max_size, len(self.df[self.df.label == i]))

        array = []
        for i in range(1, len(self.labels) + 1):
            df_minority = self.df[self.df.label == i]

            df_upsample = resample(df_minority, replace=True, n_samples=max_size, random_state=seed)
            array.append(df_upsample)

        df = pd.concat(array)
        return Kata(df.sample(frac=1), self.labels)

    def head(self):
        return self.df.head()

    def apply(self, column, callback):
        result = self.df.copy()
        result[column] = result[column].apply(callback)
        return Kata(result, self.labels)

    def filter(self, indices):
        result = self.df.copy()
        return Kata(result[indices], self.labels)

    @Experimental
    def flatten(self, column):
        columns = list(self.df.columns)
        columns.remove(column)

        df2 = self.df.set_index(columns).apply(lambda x: x.explode()).reset_index()
        df2 = df2[df2[column].notnull()]
        return Kata(df2, self.labels)

    def preprocess(self, preprocessor:Preprocessor, column: str = "value"):
        result = self.df.copy()
        result[column] = result[column].apply(preprocessor.preprocess)
        return Kata(result, self.labels)