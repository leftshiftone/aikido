import logging
from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch import cuda

from aikido.__api__ import Aikidoka, DojoKun, Kata, DojoListener, Evaluation
from aikido.__common__.iterator.BatchIterator import BatchIterator
from aikido.dojo.listener import CudaListener


class BaseDojo(ABC):

    def __init__(self, dojokun: DojoKun):
        super(BaseDojo, self).__init__()
        self.dojokun = dojokun
        self.listeners = [CudaListener()]

    def add_listener(self, listener: DojoListener):
        self.listeners.append(listener)

    def train(self, aikidoka: Aikidoka, kata: Kata):
        """
        Trains the given aikidoka with the given kata.
        If you want to get more detailed information about the training progress
        register a proper listener instance.
        """
        aikidoka.train()
        self._before_training_started(aikidoka)

        for listener in self.listeners:
            listener.training_started(aikidoka, kata, self.dojokun)

        for i in range(self.dojokun.dans):
            logging.info("Dan: {}".format(i))

            for listener in self.listeners:
                listener.dan_started(aikidoka, (i, self.dojokun.dans))

            loss, acc = self._do_dan(aikidoka, kata)

            for listener in self.listeners:
                listener.dan_finished(aikidoka, (i, self.dojokun.dans), (loss, acc))

        for listener in self.listeners:
            listener.training_finished(aikidoka, kata, self.dojokun)

    def _do_dan(self, aikidoka: Aikidoka, kata: Kata):
        total_epoch_loss = 0
        total_epoch_acc = 0

        # TODO: move iterator creation outside of loop
        iterator = self._to_iterator(aikidoka, kata)
        for i, batch in enumerate(iterator):
            for listener in self.listeners:
                listener.batch_started(aikidoka, batch, (i, len(iterator)))

            self.dojokun.optimizer.zero_grad()
            if cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)

            y_pred = aikidoka.__call__(x, batch.lengths)

            loss = self.dojokun.loss(y_pred, y)
            self._before_back_propagation(aikidoka)
            loss.backward()
            self._after_back_propagation(aikidoka)
            self.dojokun.optimizer.step()

            num_corrects = (torch.max(y_pred, 1)[1].view(y.size()).data == y.data).float().sum()
            acc = 100.0 * num_corrects / len(batch)

            for listener in self.listeners:
                listener.batch_finished(aikidoka, batch, (i, len(iterator)))

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss / len(iterator), total_epoch_acc / len(iterator)

    def _to_iterator(self, aikidoka: Aikidoka, kata: Kata) -> BatchIterator:
        embedding = aikidoka.aikidokaKun.embedding
        data = kata.apply("value", embedding.encode_ids)

        value_col = []
        label_col = []
        rowid_col = []

        for index, row in data.df.iterrows():
            for j in range(max(1, int(len(row["value"]) / self.dojokun.max_seq_len))):
                value_col.append(self.pad(row["value"][j * self.dojokun.max_seq_len:(j + 1) * self.dojokun.max_seq_len]))
                label_col.append(row["label"])
                rowid_col.append(row["rowid"])

        df = pd.DataFrame({"value": value_col, "label": label_col, "rowid": rowid_col})
        return BatchIterator(df, self.dojokun.batch_size)

    def pad(self, list):
        while (len(list) < self.dojokun.max_seq_len):
            list.append(100000) # pad
        return list


    def evaluate(self, aikidoka: Aikidoka, kata: Kata, probability: bool = True) -> Evaluation:
        """
        Evaluates the given aikidoka with the given kata. Returns a tuple containing the expected and predicted labels.
        Merges predictions with the same identifier column if the "merge" attribute is set to True.
        """
        aikidoka.eval()
        all_preds = []
        all_y = []
        all_i = []
        for _, batch in enumerate(kata):
            if torch.cuda.is_available():
                x = batch.text.cuda()
            else:
                x = batch.text
            y_pred = aikidoka(x, batch.lengths)
            predicted = torch.max(y_pred.cpu().data, 1)[1] + 1 if not probability else y_pred.cpu().data
            all_preds.extend(predicted.numpy())
            all_y.extend(batch.label.numpy())
            all_i.extend(batch.rowid.numpy())

        #return all_y, np.array(all_preds), all_i
        return Evaluation(all_y, np.array(all_preds), all_i, probability)

    def _after_back_propagation(self, aikidoka:Aikidoka):
        pass

    def _before_back_propagation(self, aikidoka:Aikidoka):
        pass

    def _before_training_started(self, aikidoka:Aikidoka):
        pass
