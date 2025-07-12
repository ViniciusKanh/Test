from collections import Counter

import numpy as np

from ._classifier import Classifier


class VotingClassifier(Classifier):
    """
    Classifier for CBA that can predict
    class labels based on a list of rules.
    """

    def __init__(self):
        super().__init__()
        self.name = "VotingClassifier"
        self.rules = None
        self.default_class = None

    def __call__(self, rules, default_class):
        self.rules = rules
        self.default_class = default_class

    def first_match(self, instance):
        class_votes = []
        for rule in self.rules:
            if rule.match(instance):
                class_votes.append(rule.consequent)

        class_votes.append(self.default_class)

        if len(class_votes) > 1:
            class_votes = np.array(class_votes).tolist()
            class_choice = Counter(class_votes).most_common(1)[0][0]
            return class_choice

        return self.default_class

    def predict(self, data):
        data = data.astype("float")
        if len(data) > 1:
            predicts = []
            for datacase in data:
                predicts.append(self.first_match(datacase))

            return predicts

        else:
            return self.first_match(data)
