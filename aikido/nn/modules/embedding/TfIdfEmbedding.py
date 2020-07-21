import logging

import torch
from torch import nn, tensor

from aikido.__api__ import Kata
from aikido.__api__.annotation import Experimental
from .AbstractEmbedding import AbstractEmbedding


@Experimental
class TfIdfEmbedding(AbstractEmbedding):

    def __init__(self, kata: Kata, min_df:int = 10):
        super().__init__()
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='utf-8',
                                         max_features=10000)#, ngram_range=(1, 3))
            features = self.tfidf.fit_transform(kata.df["value"]).toarray()

            self.features = features
            self.embeddings_ = nn.Embedding.from_pretrained(tensor(features, dtype=torch.float),
                                                            padding_idx=len(features) - 1)
            self.dim_ = len(features)
            self.vs_ = len(features)
        except ImportError:
            logging.error("-" * 100)
            logging.error("no scikit learn installation found. see https://scikit-learn.org/stable/install.html")
            logging.error("-" * 100)
            pass

    @property
    def embedding_length(self) -> int:
        return self.dim_

    @property
    def vocabulary_length(self) -> int:
        return self.vs_

    def encode_ids(self, word: str):
        def index(x):
            if x in self.tfidf.get_feature_names():
                return self.tfidf.get_feature_names().index(x)
            return 0
        return list(map(index, word.split(" ")))

    def embed(self, x):
        return self.embeddings_(x)

    def raw_embedding(self) -> nn.Embedding:
        return self.embeddings_
