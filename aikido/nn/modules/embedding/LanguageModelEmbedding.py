import logging
import torch
from torch import nn, tensor

from .AbstractEmbedding import AbstractEmbedding


# initrange=0.1
# self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
# self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
class LanguageModelEmbedding(AbstractEmbedding):

    def __init__(self, lang: str, dim: int = 300, vs: int = 100000, add_pad_emb: bool = True):
        super().__init__()
        try:
            from bpemb import BPEmb
            self.embedder = BPEmb(lang=lang, dim=dim, vs=vs, add_pad_emb=add_pad_emb)
            self.embeddings_ = nn.Embedding.from_pretrained(tensor(self.embedder.vectors, dtype=torch.float),
                                                            padding_idx=vs)
            self.dim_ = dim
            self.vs_ = vs
        except ImportError:
            logging.error("-" * 100)
            logging.error("no bpemb installation found. see https://github.com/bheinzerling/bpemb")
            logging.error("-" * 100)
            pass

    @property
    def embedding_length(self) -> int:
        return self.dim_

    @property
    def vocabulary_length(self) -> int:
        return self.vs_

    def encode_ids(self, word):
        return self.embedder.encode_ids(word)

    def embed(self, x):
        return self.embeddings_(x)

    def raw_embedding(self) -> nn.Embedding:
        return self.embeddings_
