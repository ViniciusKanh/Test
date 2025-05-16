from ._prune import Prune
import numpy as np


class OverlapPrune(Prune):

    def __init__(self):
        super().__init__()
        self.rules = None
        self.data_np = None
        self.name = "OverlapPrune"

    def __call__(self, X, y, rules):
        pertence = []
        nao_pertence = []
        y = np.random.randint(2, size=len(rules))

        cover_r = []
        for idx, r in enumerate(rules):
            cover_r2 = r.match_A(X.astype(float))
            overlap = len(set(cover_r).intersection(set(cover_r2)))/len(set(cover_r).union(set(cover_r2)))

            if overlap < 0.5:
                cover_r = cover_r2
                pertence.append(idx)
                flag_novo_nao_pertence = True

            else:
                if flag_novo_nao_pertence:
                    nao_pertence.append(idx)
                    flag_novo_nao_pertence = False

            if len(pertence) >= len(rules)*0.2:
                break

        return rules[pertence], self.get_default_class(y)

