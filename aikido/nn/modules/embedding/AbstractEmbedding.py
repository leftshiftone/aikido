from abc import abstractmethod
from torch import nn


class AbstractEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector"""
        pass

    @property
    @abstractmethod
    def vocabulary_length(self) -> int:
        """Returns the length of the vocabulary"""
        pass

    @abstractmethod
    def encode_ids(self, token:str):
        pass

    @abstractmethod
    def embed(self, x):
        pass

    def forward(self, x):
        return self.embed(x)

    def raw_embedding(self) -> nn.Embedding:
        pass
