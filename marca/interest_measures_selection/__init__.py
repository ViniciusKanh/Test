class Anova:
    """Dummy ANOVA selector used in tests."""

    def __init__(self):
        self.name = "Anova"

    def __call__(self, rules=None, measures=None):
        return measures

    def fit(self, rules, measures):
        return self

    def transform(self, rules, measures):
        return measures

    def fit_transform(self, rules, measures):
        return measures
