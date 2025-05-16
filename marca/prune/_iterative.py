from collections import Counter
from ._prune import Prune
import numpy as np
from copy import copy

from ..data_structures._car import CAR
from ..data_structures._cars import CARs


class IterativeCAR(CAR):
    def __init__(self, antecedent, consequent, support, confidence, A, B, dataset_size):
        super().__init__(antecedent, consequent, support, A, B, dataset_size)

        self.cover_A = None
        self.cover_AB = None
        self.trues = np.isnan(self.rule)

    def set_cover(self, data):
        self.cover_A = np.zeros(data.shape[0], dtype=bool)
        self.cover_AB = np.zeros(data.shape[0], dtype=bool)

        self.cover_A[self.match_A(data[:, :-1])] = True
        self.cover_AB[self.match_AB(data)] = True

    def delete_cover(self, index_covered):
        self.cover_A = np.delete(self.cover_A, index_covered, axis=0)
        self.cover_AB = np.delete(self.cover_AB, index_covered, axis=0)

    def get_A_and_AB_cover(self):
        return self.cover_A.sum(), self.cover_AB.sum()


class IterativePrune(Prune):
    def __init__(self, rank_method=None, interest_measures=None):
        super().__init__()
        self.name = "IterativePrune"
        self.rules = None
        self.data = None
        self.rank_method = rank_method
        self.interest_measures = interest_measures

    def set_rank_method(self, rank_method):
        self.rank_method = copy(rank_method)

    def set_interest_measures(self, interest_measures):
        self.interest_measures = copy(interest_measures)

    def calc_B(self, y, counter_y, rule):
        return counter_y[rule.consequent.tolist()]/y.shape[0]

    def calc_A(self, A_values, rule_index):
        return A_values[rule_index]

    def calc_AB(self, AB_values, rule_index):
        return AB_values[rule_index]

    def __call__(self, X, y, rules):
        final_classifier = CARs([])

        data = np.hstack((X.astype(float), y.astype(float).reshape(-1, 1)))
        iterative_rules = CARs.from_numpy(rules.to_numpy(), rules.support, rules.sup_antecedent, rules.sup_consequent,
                                          len(X))
        self.set_cover(data, iterative_rules)
        index = 0
        while (len(data) > 0):
            t = data[index]
            covered = False

            for r in iterative_rules:
                if r.match(t[:-1]):
                    final_classifier.append(r)
                    covered = True
                    data = np.delete(data, 0, axis=0)
                    break

            if not covered:
                index += 1

            if len(data) == 0:
                break

            new_A, new_B, new_AB = self.update_measures_values(data, iterative_rules)
            rules_remove = np.where(((new_A == 0) | (new_AB == 0)) | (new_B == 0))[0]
            iterative_rules = iterative_rules.delete(rules_remove)
            if len(iterative_rules) == 0:
                break

            new_A = np.delete(new_A, rules_remove, axis=0)
            new_B = np.delete(new_B, rules_remove, axis=0)
            new_AB = np.delete(new_AB, rules_remove, axis=0)
            iterative_rules.set_measures(
                sup_antecedent=new_A,
                sup_consequent=new_B,
                sup_rule=new_AB,
                num_transactions=len(data),
                interest_measures=self.interest_measures.get_measures(),
            )
            iterative_rules = self.rank_method(X, y, iterative_rules, self.interest_measures)

        default_class = self.get_default_class(data, y)
        return final_classifier, default_class

    def delete_cover(self, dynamic_rules, index_covered):
        for r in dynamic_rules:
            r.delete_cover(index_covered)

    def get_default_class(self, data, y):
        y_float = y.astype(float)
        if len(data) == 0:
            return Counter(y_float).most_common(1)[0][0]

        else:
            return Counter(data[:, -1]).most_common(1)[0][0]

    def set_cover(self, data, rules):
        for r in rules:
            r.set_cover(data)

    def calc_initial_cover(self, data, rules):
        cover_A = np.zeros((data.shape[0], len(rules)), dtype=bool)
        cover_AB = np.zeros((data.shape[0], len(rules)), dtype=bool)
        for idx, r in enumerate(rules):
            cover_A[r.match_A(data[:, :-1]), idx] = True
            cover_AB[r.match_AB(data), idx] = True
        return cover_A, cover_AB

    def update_measures_values(self, data, rules):
        A_values = []
        AB_values = []

        for r in rules:
            cover_A = r.cover_A.sum()
            cover_AB = r.cover_AB.sum()
            A_values.append(cover_A)
            AB_values.append(cover_AB)

        counter_y = Counter(data[:, -1])
        new_A = (np.array(A_values)/len(data)).reshape(-1, 1)
        new_B = np.array(
            [self.calc_B(data[:, -1], counter_y, rule) for rule in rules]
        ).reshape(-1, 1)
        new_AB = (np.array(AB_values)/len(data)).reshape(-1, 1)

        return new_A, new_B, new_AB
