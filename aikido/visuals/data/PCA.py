class PCA:

    def __init__(self, embedding):
        self.embedding = embedding

    def render(self, kata):
        kata2 = kata.apply("value", self.embedding.embedder.embed).df
        features = kata2["value"]

        import numpy as np
        features2 = list(filter(lambda x: len(x) > 0, features))
        x = np.array(list(map(lambda x: np.sum(x, axis=0) / x.shape[0], features2)))

        from sklearn.decomposition import PCA
        import pandas as pd

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(x)

        principalDf = pd.DataFrame(data=pca_result, columns=['pca1', 'pca2'])

        finalDf = pd.concat([principalDf, kata2[['label']]], axis = 1)
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = [1.0]
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['label'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'pca1']
                       , finalDf.loc[indicesToKeep, 'pca2']
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()