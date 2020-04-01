from aikido.__api__ import Aikidoka
from aikido.__api__ import DojoKun
from .BaseDojo import BaseDojo


class RnnDojoKun(DojoKun):
    def __init__(self, optimizer, loss, dans=20, gradient_clipping=None, batch_size: int = 64,
                 max_seq_len: int = 100):
        super().__init__(optimizer, loss, dans, batch_size, max_seq_len)
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
