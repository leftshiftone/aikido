import codecs
import json
import os
from glob import glob
from os import listdir
from os.path import isdir, join
from typing import List

import pandas as pd

from aikido.__api__ import Preprocessor
from aikido.kata import AbstractKata


class JsonKata(AbstractKata):
    """
    Kata implementation which loads the data from a json file.
    """

    def __init__(self, df, labels: List[str]):
        super().__init__(df, labels)

    """
    Load a kata instance by loading a csv file from the given filename. The kata data can be upsampled by setting 
    the 'upsample' attribute to true.
    """

    @staticmethod
    def from_folder(path: str, preprocessor: Preprocessor = Preprocessor(), encoding: str = "utf-8"):
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
            value_col.append(preprocessor.preprocess(d["text"]))
            label_col.append(labels.index(d["label"]) + 1)
            rowid_col.append(i)
            i = i + 1

        df = pd.DataFrame({"value": value_col, "label": label_col, "rowid": rowid_col})
        return JsonKata(df, labels)
