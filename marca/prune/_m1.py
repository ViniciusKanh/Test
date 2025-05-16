import collections
import numpy as np
from ..data_structures import CARs
import random
from ._prune import Prune


class M1Prune(Prune):
    def __init__(self):
        super().__init__()
        self.rules = None
        self.data_np = None
        self.name = "M1Prune"

    def __call__(self, X, y, rules):
        self.data_np = np.hstack((X, y.reshape(-1, 1))).astype(float)
        self.rules = rules

        default_classes = []

        class_distribution = collections.Counter(y)
        classdist_keys = list(class_distribution.keys())
        # r_np = self.rules.as_np()
        # empty_att_and_class = np.isnan(r_np)
        classifier_index = []
        data_search = self.data_np.copy()
        default_classes_errors = []
        rule_errors = []
        total_errors = []

        for index, rule in enumerate(self.rules):
            if data_search.shape[0] <= 0:
                break

            # rule = r_np[index]

            matched_ant = rule.match_A(data_search[:, :-1])
            matched_ant_and_consq = rule.match_AB(data_search)

            len_match = len(matched_ant)
            len_match_consq = len(matched_ant_and_consq)

            if len_match_consq > 0:
                classifier_index.append(index)
                data_search = np.delete(data_search, matched_ant, axis=0)

                # Not conventional edition
                if data_search.shape[0] == 0:
                    default_class, count_occor = None, 0

                else:
                    default_class, count_occor = collections.Counter(
                        data_search[:, -1]
                    ).most_common(1)[0]

                default_classes.append(default_class)
                rule_errors.append(len_match - len_match_consq)
                dflt_class_err = data_search.shape[0] - count_occor
                default_classes_errors.append(dflt_class_err)
                total_errors.append(dflt_class_err + sum(rule_errors))

        if len(total_errors) != 0:
            min_errors = min(total_errors)
            idx_to_cut = total_errors.index(min_errors)
            final_classifier = classifier_index[: idx_to_cut + 1]
            default_class = default_classes[idx_to_cut]
            final_classifier = self.rules[final_classifier]

        else:
            possible_default_classes = list(class_distribution)
            random_class_idx = random.randrange(0, len(possible_default_classes))
            _, default_class = classdist_keys[random_class_idx]
            final_classifier = CARs([])

        return final_classifier, default_class
