import numpy as np
from ._rank import Rank


class ElectreII:
    def __init__(
        self, c_minus=0.65, c_zero=0.75, c_plus=0.85, d_minus=0.25, d_plus=0.50
    ):
        self.weights = None
        self.criteria = None
        self.matrix_w = None
        self.matrix_s = None
        self.matrix_concordance = None
        self.matrix_discordance = None
        self.c_minus = c_minus
        self.c_plus = c_plus
        self.c_zero = c_zero
        self.d_minus = d_minus
        self.d_plus = d_plus

    def calc_concordance(self):
        self.matrix_concordance = np.array(
            [
                self.weights[np.where(self.criteria[i] >= self.criteria[j])[0]].sum()
                for i in range(self.criteria.shape[0])
                for j in range(self.criteria.shape[0])
            ]
        ).reshape(self.criteria.shape[0], self.criteria.shape[0])

    def calc_discordance(self):
        delta = np.max(self.criteria.max(axis=0) - self.criteria.min(axis=0))
        self.matrix_discordance = np.array(
            [
                np.max(self.criteria[j] - self.criteria[i]) / delta
                for i in range(self.criteria.shape[0])
                for j in range(self.criteria.shape[0])
            ]
        ).reshape(self.criteria.shape[0], self.criteria.shape[0])

    def matrix_cred(self):
        if_list = np.array(
            [
                (
                    self.matrix_concordance[i, j] >= self.matrix_concordance[j, i]
                    if i != j
                    else False
                )
                for i in range(self.matrix_concordance.shape[0])
                for j in range(self.criteria.shape[0])
            ]
        ).reshape(self.criteria.shape[0], self.criteria.shape[0])

        self.matrix_s = np.logical_and(
            np.logical_or(
                np.logical_and(
                    self.matrix_concordance >= self.c_plus,
                    self.matrix_discordance <= self.d_plus,
                ),
                np.logical_and(
                    self.matrix_concordance >= self.c_zero,
                    self.matrix_discordance <= self.d_minus,
                ),
            ),
            if_list,
        )

        self.matrix_w = np.logical_and(
            np.logical_or(
                np.logical_and(
                    self.matrix_concordance >= self.c_zero,
                    self.matrix_discordance <= self.d_plus,
                ),
                np.logical_and(
                    self.matrix_concordance >= self.c_minus,
                    self.matrix_discordance <= self.d_minus,
                ),
            ),
            if_list,
        )

    def distillation(self, ascending=False):
        if ascending:
            self.matrix_s = self.matrix_s.T
            self.matrix_w = self.matrix_w.T

        rank = []
        r_aux = []
        y = self.matrix_s
        y_w = self.matrix_w
        next_y = np.array(range(y.shape[0]))

        for k in range(10):
            d_index = np.where(~(y.any(axis=0, where=True)))[0]

            if d_index.size == 0:
                d_index = d_index

            u_index = d_index[np.where(~y_w[:, d_index].any(axis=0))[0]]

            if u_index.size == 0:
                u_index = d_index

            b_index = u_index[np.where(y_w[:, u_index].any(axis=1))[0]]

            if b_index.size == 0:
                b_index = u_index

            a = [item for item in d_index if item not in u_index]

            r = np.append(a, b_index)
            r = r.astype(int)

            r_index = next_y[r]

            rank.append(r_index)
            r_aux.extend(r_index)

            next_y = np.array(
                [i for i in range(self.matrix_s.shape[0]) if i not in r_aux]
            )

            if next_y.size == 0:
                break

            y = self.matrix_s[next_y][:, next_y]
            y_w = self.matrix_w[next_y][:, next_y]

        if ascending:
            self.matrix_s = self.matrix_s.T
            self.matrix_w = self.matrix_w.T

        return rank

    def final_rank(self, rank_ascending, rank_descending):
        rank_index_asc = np.zeros(self.criteria.shape[0])
        for idx, r in enumerate(rank_ascending):
            for item in r:
                rank_index_asc[item] = self.criteria.shape[0] - idx - 1

        rank_index_desc = np.zeros(self.criteria.shape[0])
        for idx, r in enumerate(rank_descending):
            for item in r:
                rank_index_desc[item] = idx + 1

        rank_value = rank_index_asc + rank_index_desc

        return rank_value

    def fit(self, criteria, weights):
        self.criteria = criteria
        self.weights = weights

        self.calc_concordance()
        self.calc_discordance()
        self.matrix_cred()
        rank_descending = self.distillation()
        rank_ascending = self.distillation(ascending=True)
        rank_values = self.final_rank(rank_ascending, rank_descending)

        return -rank_values


class ElectreIIRank(Rank):
    def __init__(self):
        super().__init__()
        self.name = "Electre II"

    def __call__(self, x, y, rules, measures):
        mos = rules.get_measures(normalized="rank")
        peso = np.array([1 / mos.shape[1] for _ in range(mos.shape[1])])

        rank_values = ElectreII().fit(mos, weights=peso)

        rules.sorted(by=rank_values)

        return rules
