from aikido.__api__ import Evaluation
from .AbstractVisual import AbstractVisual


class AccuracyScore(AbstractVisual):

    def render(self, evaluation: Evaluation):
        if not evaluation.isprop:
            try:
                from sklearn.metrics import accuracy_score
                print(accuracy_score(evaluation.labels, evaluation.values.flatten()))
            except ImportError:
                print("no sklearn installation found")
        else:
            try:
                import pandas as pd

                data = {'id': id, 'label': evaluation.labels, 'probs': evaluation.values.tolist()}
                df1 = pd.DataFrame(data)

                def aggregate(x):
                    result = list(map(lambda e: e.index(max(e)) + 1, x))
                    return max(result, key=result.count)

                df1 = df1.groupby(['id', 'label'], as_index=False).aggregate(aggregate)
                df2 = df1[df1.label == df1.probs]

                return len(df2) / len(df1)
            except ImportError:
                print("no pandas installation found")
