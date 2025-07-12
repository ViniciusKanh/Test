from sklearn.metrics import f1_score


class Classifier:
    def __init__(self):
        pass

    def predict(self, data):
        """Implement method of predict data"""
        pass

    def __call__(self, rules, default_class):
        """Set parameters inside classifier to predict data"""
        pass

    def score(self, x, y):
        """Return score of the classifier"""
        predicts = self.predict(x)
        return f1_score(y, predicts, average="macro")
