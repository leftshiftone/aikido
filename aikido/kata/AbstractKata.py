from typing import List

from pandas import DataFrame

from aikido.__api__ import Kata
from aikido.__api__.annotation import Experimental
from aikido.visuals.data import PCA, TSNE, WordCloud


class AbstractKata(Kata):

    def __init__(self, df: DataFrame, labels: List[str]):
        super().__init__(df, labels)

    @Experimental
    def plot_tsne(self, embedding):
        TSNE(embedding).render(self)

    @Experimental
    def plot_pca(self, embedding):
        PCA(embedding).render(self)

    @Experimental
    def plot_wordcloud(self):
        WordCloud().render(self)

    def plot_barchart(self):
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#282828'

        plt.figure(figsize=(8, 6))
        self.df.groupby('label').label.count().plot.bar(ylim=0)
        plt.show()