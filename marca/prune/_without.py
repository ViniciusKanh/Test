from collections import Counter

import numpy as np

from ._prune import Prune


class WithoutPrune(Prune):
    def __init__(self):
        super().__init__()
        self.name = "WithoutPrune"

    def get_default_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def __call__(self, X, y, rules):
        return rules, self.get_default_class(y.astype(float))
