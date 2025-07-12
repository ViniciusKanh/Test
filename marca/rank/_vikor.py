from ._rank import Rank
import numpy as np


class Vikor:
    def __init__(self, v):
        self.criteria = None
        self.weights = None
        self.v = v

    def fit(self, criteria, weights):
        f_max = np.max(criteria, axis=0)
        f_min = np.min(criteria, axis=0)

        s_i = np.sum(
            weights * (np.nan_to_num((f_max - criteria) / (f_max - f_min))), axis=1
        )
        r_i = np.max(
            weights * (np.nan_to_num((f_max - criteria) / (f_max - f_min))), axis=1
        )

        s_star = np.min(s_i)
        s_min = np.max(s_i)

        r_star = np.min(r_i)
        r_min = np.max(r_i)

        q_i = (self.v * (s_i - s_star) / (s_min - s_star)) + (
            (1 - self.v) * (r_i - r_star) / (r_min - r_star)
        )

        # dq = 1 / (criteria.shape[0] - 1)

        return -q_i


class VikorRank(Rank):
    def __init__(self, measures=None, name="Vikor", v=0.5):
        super().__init__()
        self.name = name
        self.v = v
        self.measures = measures

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(
            measures=measures, normalized="rank"
        )
        weights = measures.get_weights()

        rank_values = Vikor(v=self.v).fit(interest_measures_values, weights)
        return rank_values


if __name__ == "__main__":
    by = Vikor(0.1).fit(
        np.array([[1, 1, 2, 4], [2, 2, 2, 5], [5, 2, 5, 6]]), [1, 1, 1, 1]
    )

    print(by)

    r = np.array(
        sorted(
            enumerate(["R1", "R02", "R003"]),
            key=lambda x: (by[x[0]], len(x)),
            reverse=True,
        )
    )[:, 1].tolist()

    print(r)
