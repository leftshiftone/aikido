from typing import List

import pandas as pd

from aikido.__api__ import Preprocessor
from aikido.__api__.Kata import Kata


class CsvKata(Kata):
    """
    Kata implementation which loads the data from a csv file.
    """

    def __init__(self, df, labels: List[str]):
        super().__init__(df, labels)

    """
    Load a kata instance by loading a csv file from the given filename. The kata data can be upsampled by setting 
    the 'upsample' attribute to true.
    """

    @staticmethod
    def from_file(path: str, preprocessor: Preprocessor = Preprocessor(), delimiter: str = ";", encoding: str = "utf-8"):

        # parse labels
        # ************
        labels = set()
        with open(path, 'r', encoding=encoding) as datafile:
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

        with open(path, 'r', encoding=encoding) as datafile:
            data = [line.strip().split(delimiter, maxsplit=1) for line in datafile]
            i = 0
            for row in data:
                value_col.append(preprocessor.preprocess(row[1]))
                label_col.append(labels.index(row[0]) + 1)
                rowid_col.append(i)
                i = i + 1

        df = pd.DataFrame({"value": value_col, "label": label_col, "rowid": rowid_col})
        return CsvKata(df, labels)
