import numpy as np
from sklearn.feature_selection import mutual_info_classif
class IGSelector:
    def __init__(self, ratio: float = 0.5):
        self.name = "InformationGain"
        self.ratio = ratio
        self.selected_indices = None

    def fit(self, X, y):
        X = np.asarray(X)
        scores = mutual_info_classif(X, y)
        k = max(1, int(self.ratio * X.shape[1]))
        self.selected_indices = np.argsort(scores)[::-1][:k]
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)