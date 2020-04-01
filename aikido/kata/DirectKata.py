from aikido.__api__.Kata import Kata


class DirectKata(Kata):
    """
    Kata implementation which loads the data from an csv file.
    """

    def __init__(self, df, batch_size: int, clipping: int):
        super().__init__(df, batch_size, clipping)

    """
    Load a kata instance by using the given dataframe. The kata data can be upsampled by setting the 'upsample'
    attribute to true.
    """

    @staticmethod
    def from_dataframe(df, upsample: bool, seed: int, batch_size: int = 64, clipping: int = 100):
        labels = df[df.labels]
        df = df.sample(frac=1) if upsample is not True else DirectKata._upsample(df, labels, seed)
        return DirectKata(df, batch_size, clipping)
