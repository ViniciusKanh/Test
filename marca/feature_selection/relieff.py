# marca/feature_selection/relieff.py
import numpy as np
from skrebate import ReliefF

class ReliefFSelector:
    def __init__(self, ratio: float = 0.5, n_neighbors: int = 100):
        self.name = "ReliefF"
        self.ratio = ratio
        self.n_neighbors = n_neighbors
        self.selected_indices = None
        self._estimator = ReliefF(n_neighbors=self.n_neighbors)

    def fit(self, X, y):
        X = X.astype(float)
        self._estimator.fit(X, y)
        scores = self._estimator.feature_importances_
        k = max(1, int(self.ratio * X.shape[1]))
        self.selected_indices = np.argsort(scores)[::-1][:k]
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
