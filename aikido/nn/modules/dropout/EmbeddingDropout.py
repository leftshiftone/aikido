import torch
from torch import Tensor, LongTensor
from typing import Optional, Collection

from aikido.nn.modules.embedding import AbstractEmbedding


class EmbeddingDropout(torch.nn.Module):
    """Apply dropout with probability 'prop' to an embedding layer 'emb'"""

    def __init__(self, emb: AbstractEmbedding, prob: float = 0.1):
        super().__init__()
        self.emb = emb
        self.prob = prob
        self.pad_idx = self.emb.raw_embedding().padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, x: LongTensor, scale: Optional[float] = None) -> Tensor:
        emb: torch.nn.Embedding = self.emb.raw_embedding()

        if self.training and self.prob != 0:
            size = (emb.weight.size(0), 1)
            mask = self.dropout_mask(emb.weight.data, size, self.prob)
            masked_emb = emb.weight * mask
        else:
            masked_emb = emb.weight

        if scale: masked_emb.mul_(scale)
        return torch.nn.functional.embedding(x,
                                             masked_emb,
                                             self.pad_idx,
                                             emb.max_norm,
                                             emb.norm_type,
                                             emb.scale_grad_by_freq,
                                             emb.sparse
                                             )

    def dropout_mask(self, x: Tensor, sz: Collection[int], p: float):
        "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
        return x.new(*sz).bernoulli_(1 - p).div_(1 - p)
