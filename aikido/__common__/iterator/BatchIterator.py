import torch


class Batch:
    def __init__(self, rowid, value, label, lengths):
        self.rowid = rowid
        # TODO: rename to value
        self.text = value
        self.label = label
        # TODO: rename to length
        self.lengths = lengths

    def __len__(self):
        return self.text.shape[1]


class BatchIterator:

    def __init__(self, df, batch_size, max_seq_len):
        self.df = df
        self.len = len(df)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return int((self.len / self.batch_size) + (1 if self.len % self.batch_size is not 0 else 1))

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if (self.counter * self.batch_size) < (self.len - 1):
            index1 = self.counter * self.batch_size
            index2 = min((self.counter + 1) * self.batch_size, self.len)

            self.counter += 1
            return self.to_tensor(self.df[index1:index2])
        else:
            raise StopIteration

    def to_tensor(self, df):
        def txtLen(text):
            return min(len(text), self.max_seq_len)
        def slice(list):
            return self.pad(list[0: self.max_seq_len])

        value = torch.LongTensor(list(map(slice, df.value.values.tolist()))).t()
        label = torch.LongTensor(df.label.values.tolist())
        rowid = torch.LongTensor(df.rowid.values.tolist())
        lengths = torch.LongTensor(df.value.map(txtLen).values.tolist())

        return Batch(rowid, value, label, lengths)

    def pad(self, list):
        while (len(list) < self.max_seq_len):
            list.append(100000) # pad
        return list
