import numpy as np
import pandas as pd
from sklearn.utils import resample

from aikido.__api__ import Preprocessor
from aikido.__api__.Kata import Kata
from aikido.nn.modules.embedding import AbstractEmbedding


class CsvKata(Kata):
    """
    Kata implementation which loads the data from an csv file.
    """

    def __init__(self, df, batch_size: int, clipping: int):
        super().__init__(df, batch_size, clipping)

    """
    Load a kata instance by using the given dataframe. The kata data can be upsampled by setting the 'upsample'
    attribute to true.
    """

    """
    Load a kata instance by loading a csv file from the given filename. The kata data can be upsampled by setting 
    the 'upsample' attribute to true.
    """

    @staticmethod
    def from_file(filename: str, embedder:AbstractEmbedding, preprocessor: Preprocessor = Preprocessor(), delimiter: str = ";",
                  encoding: str = "utf-8", seed: int = 123, upsample: bool = True, batch_size: int = 64,
                  max_text_len: int = 100):

        # parse labels
        # ************
        labels = set()
        with open(filename, 'r', encoding=encoding) as datafile:
            data = [line.strip().split(delimiter, maxsplit=1) for line in datafile]
            for entry in data:
                labels.add(entry[0])

        labels = list(labels)
        labels.sort()

        # parse values
        # ************
        value_col = []
        label_col = []
        rowid_col = []

        with open(filename, 'r', encoding=encoding) as datafile:
            data = [line.strip().split(delimiter, maxsplit=1) for line in datafile]
            i = 0
            for row in data:
                text = preprocessor.preprocess(row[1])
                ids = embedder.encode_ids(text)
                for j in range(max(1, int(len(ids) / max_text_len))):
                    value_col.append(ids[j * max_text_len:(j + 1) * max_text_len])
                    label_col.append(labels.index(row[0]) + 1)
                    rowid_col.append(i)
                i = i + 1

        df = pd.DataFrame({"value": value_col, "label": label_col, "rowid": rowid_col})
        df = df.sample(frac=1) if upsample is not True else CsvKata._upsample(df, labels, seed)

        return CsvKata(df, batch_size, max_text_len)

    @staticmethod
    def _upsample(df, labels, seed: int):
        max_size = 0

        for i in range(1, len(labels) + 1):
            max_size = max(max_size, len(df[df.label == i]))

        array = []
        for i in range(1, len(labels) + 1):
            df_minority = df[df.label == i]
            df_upsample = resample(df_minority, replace=True, n_samples=max_size, random_state=seed)
            array.append(df_upsample)

        df = pd.concat(array)
        return df.sample(frac=1)

    def split(self, ratio: float):
        """
        Splits this kata into two separate katas considering the split ratio.
        """
        (df, batch_size, clipping) = self.unapply()
        df1, df2 = np.split(df.sample(frac=1), [int(ratio * len(df))])
        return CsvKata(df1, batch_size, clipping), CsvKata(df2, batch_size, clipping)
