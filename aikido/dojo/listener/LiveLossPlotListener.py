import logging

from aikido.__api__.Aikidoka import Aikidoka
from aikido.__api__.Dojo import DojoListener, DojoKun
from aikido.__api__.Kata import Kata

try:
    from livelossplot import PlotLosses
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#282828'
except ImportError:
    logging.error("-" * 100)
    logging.error("no livelossplot installation found. see https://pypi.org/project/livelossplot/")
    logging.error("-" * 100)
    pass


class LiveLossPlotListener(DojoListener):
    """
    DojoListener implementation which renders a livelossplot after finishing a dan.
    """

    def __init__(self):
        self.liveloss = None

    def training_started(self, aikidoka: Aikidoka, kata: Kata, kun: DojoKun):
        self.liveloss = PlotLosses()

    def dan_finished(self, aikidoka: Aikidoka, run: (int, int), metrics: (float, float)):
        (loss, acc) = metrics

        self.liveloss.update({"loss": loss, "train_acc": acc})
        self.liveloss.draw()
