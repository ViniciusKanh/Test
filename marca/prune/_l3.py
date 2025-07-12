import numpy as np
from marca.data_structures import CARs

from ._prune import Prune


class L3Prune(Prune):
    def __init__(self):
        super().__init__()
        self.rules = None
        self.data_np = None
        self.name = "L3Prune"

    def __call__(self, X, y, rules):
        data = np.hstack((X, y.reshape(-1, 1))).astype(float)
        # selected_rules = CARs([])

        for r in rules:
            r.right = 0
            r.wrong = 0
            r.dataClassified = []

        while (len(data) > 0) and (len(rules) > 0):

            to_delete = []

            for idx in range(len(data)):
                d = data[idx]
                covered = False
                nr = len(rules)

                if len(rules) == 0:
                    break

                # while (not covered) and (nr > 0):
                for r in rules:
                    if r.match_A(d[:-1]):
                        r.dataClassified.append(d)
                        covered = True

                        if r.consequent == d[-1]:
                            r.right += 1

                        else:
                            r.wrong += 1

                    if covered:
                        break

                to_delete.append(idx)

            data = np.delete(data, to_delete, axis=0)

            for rule in rules:
                if (rule.wrong > 0) and (rule.right == 0):
                    rules.remove(rule)
                    data = np.concatenate((data, rule.dataClassified))

        usedRules = CARs([])
        spareRules = CARs([])

        for rule in rules:
            if rule.right > 0:
                usedRules.append(rule)

            else:
                spareRules.append(rule)

        default_class = self.get_default_class(y)

        return usedRules, default_class
