import pandas as pd

from aikido.__api__.Kata import Kata


class EmptyKata(Kata):
    """
    Empty Kata implementation used by models which do not need training by a kata.
    """

    def __init__(self):
        super().__init__(pd.DataFrame({}), [])
