import logging


class TSNE:

    def __init__(self, embedding):
        self.embedding = embedding

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logging.error("-" * 100)
            logging.error("no scikit learn installation found. see https://scikit-learn.org/stable/install.html")
            logging.error("-" * 100)
            pass


    def render(self, kata):
        from sklearn.manifold import TSNE
        from matplotlib import pyplot as plt
        import numpy as np

        kata2 = kata.apply("value", self.embedding.embedder.embed).df
        features = kata2["value"]
        labels = kata2["label"]

        features2 = list(filter(lambda x: len(x) > 0, features))
        x = np.array(list(map(lambda x: np.sum(x, axis=0) / x.shape[0], features2)))

        tsne = TSNE(n_components=2, random_state=0, verbose=0, perplexity=25, n_iter=300)

        X_2d = tsne.fit_transform(x[500:1000])

        # TODO: style
        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels[500:1000])