from aikido.__api__ import Evaluation
from .AbstractVisual import AbstractVisual


try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    print("no matplotlib installation found")
try:
    import seaborn as sns
except ImportError:
    print("no seaborn installation found")
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    print("no sklearn installation found")


class ConfusionMatrix(AbstractVisual):

    def __init__(self, figsize=(10, 10)):
        super().__init__()
        self.figsize = figsize

    def render(self, evaluation:Evaluation):
        if evaluation.isprop:
            raise ValueError("evaluation 'isprop' must be False")

        conf_mat = confusion_matrix(evaluation.labels, evaluation.values)
        plt.subplots(figsize=self.figsize)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    cmap=ListedColormap(['#282828', '#192346', '#055087', '#055569', '#058296', '#0A96A0', '#0FBEC8', '#05E1B9', '#0FF5C8']),
                    xticklabels=evaluation.labels,
                    yticklabels=evaluation.labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
