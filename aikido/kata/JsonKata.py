import codecs
import json
import os
from glob import glob
from os import listdir
from os.path import isdir, join
from typing import List

import numpy as np
import pandas as pd
from sklearn.utils import resample

from aikido.__api__ import Preprocessor
from aikido.__api__.Kata import Kata
from aikido.nn.modules.embedding import AbstractEmbedding


class JsonKata(Kata):
    """
    Kata implementation which loads the data from a json file.
    """

    def __init__(self, df, batch_size: int, clipping: int, labels: List[str]):
        super().__init__(df, batch_size, clipping, labels)

    """
    Load a kata instance by using the given dataframe. The kata data can be upsampled by setting the 'upsample'
    attribute to true.
    """

    """
    Load a kata instance by loading a csv file from the given filename. The kata data can be upsampled by setting 
    the 'upsample' attribute to true.
    """

    @staticmethod
    def from_folder(path: str, embedder:AbstractEmbedding, preprocessor: Preprocessor = Preprocessor(),
                  encoding: str = "utf-8", seed: int = 123, upsample: bool = True, batch_size: int = 64,
                  max_text_len: int = 100):

        # parse labels
        # ************
        labels = [f for f in listdir(path) if isdir(join(path, f))]
        result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.json'))]

        # parse values
        # ************
        value_col = []
        label_col = []
        rowid_col = []

        i = 0
        for file in result:
            d = json.load(codecs.open(file, 'r', encoding))
            ids = embedder.encode_ids(d["text"])
            for j in range(max(1, int(len(ids) / max_text_len))):
                value_col.append(ids[j * max_text_len:(j + 1) * max_text_len])
                label_col.append(labels.index(d["label"]) + 1)
                rowid_col.append(i)
            i = i + 1

        df = pd.DataFrame({"value": value_col, "label": label_col, "rowid": rowid_col})
        df = df.sample(frac=1) if upsample is not True else JsonKata._upsample(df, labels, seed)

        print(df.shape)

        return JsonKata(df, batch_size, max_text_len, labels)

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
        return JsonKata(df1, batch_size, clipping, self.labels), JsonKata(df2, batch_size, clipping, self.labels)
