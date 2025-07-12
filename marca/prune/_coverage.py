import numpy as np
from ..data_structures import CARs
from ._prune import Prune
from collections import Counter


class CoveragePrune(Prune):
    def __init__(self, top_k):
        super().__init__()
        self.name = "CoveragePrune"
        self.top_k = top_k

    def set_default_class(self, y_datacases):
        return Counter(y_datacases).most_common(1)[0][0]

    def __call__(self, X, y, rules):
        datacases = X.astype(float)
        y_datacases = y.astype(float)

        r_np = rules.antecedent
        c_np = rules.consequent
        t_np = np.isnan(r_np)

        instances_covered = np.zeros(len(datacases))
        rules_cover = np.zeros(len(r_np))

        indice = 0
        while (indice < len(rules)) and ((instances_covered < self.top_k).any()):
            regra = r_np[indice]
            classe = c_np[indice]
            empty = t_np[indice]

            for indice_dataset in np.where(instances_covered < self.top_k)[0]:
                atributos = datacases[indice_dataset]
                matched_rule = np.logical_or(empty, np.equal(regra, atributos)).all()

                if matched_rule:
                    instances_covered[indice_dataset] += 1
                    rules_cover[indice] += 1

            indice += 1

        final_classifier = CARs()

        for indice in np.where(rules_cover != 0)[0]:
            final_classifier.append(rules[indice])

        instances_not_full_covered = np.where(instances_covered < self.top_k)[0]

        if len(instances_not_full_covered) == 0:
            default_class = self.set_default_class(y_datacases)

        else:
            default_class = self.set_default_class(
                y_datacases[instances_not_full_covered]
            )

        return final_classifier, default_class
