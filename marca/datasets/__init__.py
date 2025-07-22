import numpy as np
from sklearn.datasets import load_iris as _load_iris
from sklearn.model_selection import StratifiedKFold


def load_iris(fold=0, n_splits=10, random_state=42):
    """Return train and test splits of the iris dataset for a given fold."""
    X, y = _load_iris(return_X_y=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if idx == fold:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            return X_train, y_train, X_test, y_test
    raise ValueError("Fold index out of range")
