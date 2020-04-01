from abc import ABC
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
# FIXME: scikitlearn dependency
from sklearn.utils import resample


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

    def plot_barchart(self):
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#282828'

        plt.figure(figsize=(8, 6))
        self.df.groupby('label').label.count().plot.bar(ylim=0)
        plt.show()

    # https://www.datacamp.com/community/tutorials/wordcloud-python
    def plot_tagcloud(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wordcloud = WordCloud().generate(["text"])

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
