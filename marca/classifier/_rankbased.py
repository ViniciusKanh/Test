from collections import Counter
import numpy as np
from ._classifier import Classifier


class RankBasedClassifier(Classifier):
    """
    RankBasedClassifier is a classifier that uses the rank method to predict the class of an instance.
    Adapted from:
    Ayyat, S., Lu, J., & Thabtah, F.A. (2014). Class Strength Prediction Method for Associative Classification.
    """
    def __init__(self):
        super().__init__()
        self.name = "RankBasedClassifier"
        self.rules = None
        self.default_class = None

    def __call__(self, rules, default_class):
        self.rules = rules
        self.default_class = default_class

    def calc_score(self, matched_rules, matched_class):
        number_rules = Counter(matched_rules.consequent)
        counter_score = number_rules.copy()

        idx = matched_class.shape[0] + 1
        for klass in matched_rules.consequent:
            counter_score[klass] += idx
            idx -= 1

        score_rules = np.array(
            [
                klass
                for klass, score in counter_score.items()
                if score == max(counter_score.values())
            ]
        )

        if len(score_rules) > 1:
            number_rules_tie = np.array([number_rules[n] for n in score_rules])
            max_number_rules = np.where(number_rules_tie == np.max(number_rules_tie))[0]

            if max_number_rules.shape[0] > 1:
                np.random.seed(42)
                predict_class = np.random.choice(score_rules[max_number_rules])
            else:
                predict_class = score_rules[max_number_rules][0]

        else:
            predict_class = score_rules[0]

        return predict_class

    def predict_instance(self, x):
        index_rules = self.rules.antecedent_match(x)
        matched_rules = self.rules[index_rules]

        if len(matched_rules) != 0:
            matched_class = np.unique(matched_rules.consequent)
            if len(matched_class) == 1:
                return matched_class[0]

            else:
                return self.calc_score(matched_rules, matched_class)

        else:
            return self.default_class

    def predict(self, data):
        predicted = []
        instances = data.astype(float)
        for x in instances:
            predicted.append(self.predict_instance(x))

        return predicted
