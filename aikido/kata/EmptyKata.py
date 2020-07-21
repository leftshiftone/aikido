import pandas as pd

from aikido.kata import AbstractKata


class EmptyKata(AbstractKata):
    """
    Empty Kata implementation used by models which do not need training by a kata.
    """

    def __init__(self):
        super().__init__(pd.DataFrame({}), [])
