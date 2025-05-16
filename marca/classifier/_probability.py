import numpy as np
from ._classifier import Classifier


class ProbabilityClassifier(Classifier):
    def __init__(self, top_k):
        self.name = "ProbabilityClassifierTop" + str(top_k)
        self.rules = None
        self.default_class = None
        self.top_k = top_k

    def __call__(self, rules, default_class):
        self.rules = rules
        self.default_class = default_class

        self.y_uniques = np.unique(self.rules.consequent)
        self.L = self.y_uniques.shape[0]

        self.r_np = self.rules.antecedent
        self.c_np = self.rules.consequent
        self.empty = np.isnan(self.r_np)

    def predict(self, data):
        if len(data) > 1:
            predicts = []
            rules_ids = []
            for instance in data:
                predicts.append(self.predict_instance(instance))

            return predicts

        else:
            return self.predict_instance(data)

    def predict_instance(self, instance):
        datacase_test = instance.astype(float)
        index_matched = np.where((self.empty | (self.r_np == datacase_test)).all(1))[0]
        # regras_matched = self.r_np[index_matched][:self.top_k]
        classes_matched = self.c_np[index_matched][
            : min(self.c_np[index_matched].shape[0], self.top_k)
        ]
        measure_to_rank = self.rules.get_rank_values()

        if len(classes_matched) == 0:
            return self.default_class

        if (len(classes_matched) == 1) or (len(np.unique(classes_matched)) == 1):
            return classes_matched[0]

        else:
            return self.calc_classe(classes_matched, measure_to_rank)

    def calc_classe(self, classes_matched, interet_measures):
        p_c = interet_measures
        p_w = (1 - interet_measures) / (self.L - 1)

        probs = []
        for possivel_y in self.y_uniques:
            prob = np.sum(p_c[np.where(possivel_y == classes_matched)[0]])
            prob += np.sum(p_w[np.where(possivel_y != classes_matched)[0]])

            probs.append(prob)

        classe_pred = self.y_uniques[np.argmax(probs / np.sum(probs))]
        return classe_pred
