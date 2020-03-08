from abc import abstractmethod
from torch import nn

from aikido.__api__.Dojo import Evaluation


class AbstractVisual(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def render(self, evaluation: Evaluation):
        """Renders the visual"""
        pass
