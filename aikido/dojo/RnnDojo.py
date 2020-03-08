from aikido.__api__ import Aikidoka
from aikido.__api__ import DojoKun
from .BaseDojo import BaseDojo


class RnnDojoKun(DojoKun):
    def __init__(self, optimizer, loss, dans=20, gradient_clipping=None):
        super().__init__(optimizer, loss, dans)
        self.gradient_clipping = gradient_clipping


class RnnDojo(BaseDojo):

    def __init__(self, dojokun: RnnDojoKun):
        super().__init__(dojokun)

    def _after_back_propagation(self, aikidoka: Aikidoka):
        # perform gradient clipping
        if self.dojokun.gradient_clipping is not None:
            params = list(filter(lambda p: p.grad is not None, aikidoka.parameters()))
            for p in params:
                p.grad.data.clamp_(-self.dojokun.gradient_clipping, self.dojokun.gradient_clipping)
