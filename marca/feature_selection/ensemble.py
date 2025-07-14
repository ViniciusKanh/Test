import numpy as np
from .information_gain import IGSelector
from .relieff import ReliefFSelector

class EnsembleSelector:
    def __init__(self, ratio: float = 0.5):
        self.name = "EnsembleIGReliefF"
        self.ratio = ratio
        self.ig = IGSelector(ratio)
        self.relf = ReliefFSelector(ratio)
        self.selected_indices = None

    def fit(self, X, y):
        self.ig.fit(X, y)
        self.relf.fit(X, y)
        self.selected_indices = np.union1d(
            self.ig.selected_indices, self.relf.selected_indices
        ).astype(int)
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
