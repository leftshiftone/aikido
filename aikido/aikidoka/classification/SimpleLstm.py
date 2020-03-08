import torch
import torch.nn.functional as F
from torch import nn

from aikido.__api__.Aikidoka import Aikidoka
from aikido.nn.modules.dropout import EmbeddingDropout
from aikido.nn.modules.dropout import WeightDropout
from aikido.nn.modules.embedding import AbstractEmbedding


class SimpleLstm(Aikidoka):
    """
    Simple LSTM model.

    Implementation of a standard LSTM with regularization using PyTorch for text classification.
    Regularization can be defined by setting the options like
    * embedding dropout
    * weight drop
    """

    def __init__(self, aikidokaKun: 'SimpleLstmKun'):
        super().__init__()
        self.aikidokaKun = aikidokaKun

        self.embeddings = aikidokaKun.embedding

        self.lstm = nn.LSTM(input_size=aikidokaKun.embedding.embedding_length,
                            hidden_size=aikidokaKun.hidden_size,
                            num_layers=aikidokaKun.hidden_layers,
                            dropout=aikidokaKun.dropout,
                            bidirectional=aikidokaKun.bidirectional)

        self.encoder = EmbeddingDropout(aikidokaKun.embedding, self.aikidokaKun.embedding_dropout)
        self.lstm = WeightDropout(self.lstm, prob=aikidokaKun.weight_dropout)
        self.dropout = nn.Dropout(aikidokaKun.dropout)
        self.fc1 = nn.Linear(2 * aikidokaKun.hidden_size, aikidokaKun.hidden_size)
        self.fc2 = nn.Linear(aikidokaKun.hidden_size, aikidokaKun.output_size)

        self.setDevice()

    def forward(self, x, x_len):
        # shape: (max_sen_len, batch_size)
        x = self.encoder(x)
        # shape: (max_sen_len=20, batch_size=64,embed_size=300)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, enforce_sorted=False)
        x, (_, _) = self.lstm(x)  # shape: seq_length, batch_size, hidden_size * directions
        x, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(x)
        # hidden_layers * 2, batch_size, hidden_size

        x = F.relu(torch.transpose(x, 0, 1).transpose(1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


class SimpleLstmKun(object):
    def __init__(self,
                 embedding:AbstractEmbedding,
                 hidden_layers: int = 2,
                 hidden_size: int = 64,
                 bidirectional: bool = True,
                 output_size: int = 9,
                 dropout: float = 0.4,
                 embedding_dropout: float = 0.2,
                 weight_dropout: float = 0.2):
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.weight_dropout = weight_dropout
        self.embedding = embedding
