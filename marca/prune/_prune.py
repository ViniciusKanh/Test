from collections import Counter


class Prune:
    def __init__(self, **kwargs):
        self.interest_measures = None
        self.name = None

    def __str__(self):
        return self.name

    def set_measures(self, measures):
        self.interest_measures = measures

    def __call__(self, X, y, rules):
        pass

    def get_default_class(self, y):
        y_float = y.astype(float)
        return Counter(y_float).most_common(1)[0][0]
