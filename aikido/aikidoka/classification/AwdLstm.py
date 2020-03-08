import torch
import torch.nn.functional as F
from torch import nn

from aikido.__api__.Aikidoka import Aikidoka
from aikido.nn.modules.dropout import EmbeddingDropout
from aikido.nn.modules.dropout import RNNDropout
from aikido.nn.modules.dropout import WeightDropout
from aikido.nn.modules.embedding import AbstractEmbedding


class AwdLstm(Aikidoka):
    """
    Averaged Stochastic Gradient Descent Weight-Dropped LSTM model inspired by https://arxiv.org/abs/1708.02182.
    """

    def __init__(self, aikidokaKun: 'AwdLstmKun'):
        super().__init__()

        self.aikidokaKun = aikidokaKun
        self.encoder = EmbeddingDropout(aikidokaKun.embedding, self.aikidokaKun.embedding_dropout)

        self.rnns = self._prepare_rnn(aikidokaKun)
        self.hidden_dps = nn.ModuleList([RNNDropout(aikidokaKun.dropout) for l in range(aikidokaKun.rnn_layers)])

        self.fc1 = nn.Linear(2 * aikidokaKun.hidden_size, aikidokaKun.hidden_size)
        self.fc2 = nn.Linear(aikidokaKun.hidden_size, aikidokaKun.output_size)

    def _prepare_rnn(self, aikidokaKun:'AwdLstmKun'):
        emb_size = aikidokaKun.embedding.embedding_length

        rnns = [nn.LSTM(
            input_size=emb_size if layer == 0 else aikidokaKun.hidden_size,
            hidden_size=aikidokaKun.hidden_size,
            num_layers=aikidokaKun.hidden_layers,
            dropout=aikidokaKun.dropout,
            bidirectional=aikidokaKun.bidirectional
        ) for layer in range(aikidokaKun.rnn_layers)]
        rnns = [WeightDropout(rnn, aikidokaKun.weight_dropout) for rnn in rnns]
        return nn.ModuleList(rnns)

    def forward(self, x, x_len):
        # shape: (max_sen_len, batch_size)
        x = self.encoder(x)
        # shape: (max_sen_len=20, batch_size=64,embed_size=300)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, enforce_sorted=False)

        hidden = None
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            x, hidden = rnn(x, hidden)
            if l != self.aikidokaKun.rnn_layers - 1 : x = hid_dp(x)

        # shape: seq_length, batch_size, hidden_size * directions
        x, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(x)
        # shape: hidden_layers * 2, batch_size, hidden_size

        x = F.relu(torch.transpose(x, 0, 1).transpose(1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


class AwdLstmKun(object):
    def __init__(self,
                 embedding:AbstractEmbedding,
                 hidden_layers: int = 2,
                 hidden_size: int = 64,
                 bidirectional: bool = True,
                 output_size: int = 9,
                 dropout: float = 0.4,
                 embedding_dropout: float = 0.2,
                 weight_dropout: float = 0.2,
                 rnn_layers: int = 1):
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.weight_dropout = weight_dropout
        self.embedding = embedding
        self.rnn_layers = rnn_layers
