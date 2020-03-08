from aikido.__api__.Aikidoka import Aikidoka
from aikido.__api__.Dojo import DojoListener


# TODO: https://github.com/huggingface/transformers/blob/34a3c25a3068ab5cdbecb08ddf2866f1209fd2dd/src/transformers/optimization.py#L47
class LearningRateStepListener(DojoListener):
    """
    DojoListener implementation which optimizes the learning rate via a scheduler.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def dan_finished(self, aikidoka: Aikidoka, run: (int, int), metrics: (float, float)):
        self.scheduler.step()
