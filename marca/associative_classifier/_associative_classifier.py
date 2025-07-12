import numpy as np
from sklearn.metrics import f1_score


class AssociativeClassifier:
    def __init__(self):
        self.time = None
        self.rules_pruned = None
        self.rules = None
        self.default_class = None
        self.classifier = None

    def fit(self, X, y):
        pass

    def score(self, X, y):
        y_predicted = self.predict(X)
        f1 = f1_score(
            y_true=y.astype(int).astype(str),
            y_pred=np.array(y_predicted).astype(int).astype(str),
            average="macro",
        )
        return f1

    def predict(self, X):
        return self.classifier.predict(X)

    def get_metrics(self, data, metrics):
        result_metrics = {}
        if "overlap" in metrics:
            result_metrics["overlap"] = self.get_overlap(data)

        if "time" in metrics:
            result_metrics["time"] = self.get_time()

        if "size" in metrics:
            result_metrics["size"] = self.get_size()

        if "length" in metrics:
            result_metrics["length"] = self.get_length()

        return result_metrics

    def get_size(self):
        return len(self.rules_pruned)

    def get_length(self):
        return np.mean(
            [np.isnan(self.rules_pruned.antecedent).sum(axis=1).mean()]
        )

    def get_time(self):
        return np.sum(list(self.time.values()))

    def get_overlap(self, data):
        if (len(self.rules_pruned) == 1) or (len(self.rules_pruned) == 0):
            return 1

        index_intersect = []
        for i in range(len(self.rules_pruned)):
            index_intersect.append(set(self.rules_pruned[i].match_A(data)))

        overlap = np.zeros((len(self.rules_pruned), len(self.rules_pruned)))
        for i in range(len(self.rules_pruned)):
            for j in range(len(self.rules_pruned)):
                if i < j:
                    overlap[i][j] = len(
                        index_intersect[i].intersection(index_intersect[j])
                    )

        overlap = overlap / len(data)
        overlap_value = (
            2 / (len(self.rules_pruned) * (len(self.rules_pruned) - 1))
        ) * overlap.sum().sum()
        return overlap_value
