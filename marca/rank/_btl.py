from numba import jit, prange
import numpy as np
from ._rank import Rank

np.seterr(divide="ignore", over="ignore", invalid="ignore")


@jit(
    ["int64[:](float64[:,:])"],
    nopython=True,
    parallel=True,
)
def generate_comparisons(measures):
    n = measures.shape[0]
    aux = np.empty(n * n, dtype=np.int64)
    for i in prange(n):
        for j in prange(n):
            aux[i * n + j] = (measures[i] > measures[j]).sum()
    return aux


class BTL:
    def __init__(self, iterations=100):
        self.iterations = iterations

    def generate_comparison_old(self, measures):
        df = np.zeros((len(measures), len(measures)))
        for i in range(len(measures)):
            for j in range(len(measures)):
                df[i][j] = sum(measures[i] > measures[j])
        return df

    def get_p(self, measures):
        w = measures.sum(axis=1)
        w_sum = measures + measures.T

        p = np.ones((len(measures), len(measures)))

        for _ in range(self.iterations):
            # print(p, w)
            # Need fix in p value to avoid division by zero

            p = w / (w_sum / (p + p.T)).sum(axis=1)
            p = p / p.sum()
            p = np.array([p] * len(p))

        return p[0]

    def fit(self, measures):
        df = generate_comparisons(measures)
        df = df.reshape(measures.shape[0], measures.shape[0])
        p = self.get_p(df)

        return p.tolist()


class BTLRank(Rank):
    def __init__(self, name="", iterations=1):
        super().__init__()
        self.name = "BTL" + name
        self.iterations = iterations

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(
            measures=measures, normalized="rank"
        )
        rank_values = BTL(iterations=self.iterations).fit(interest_measures_values)
        return rank_values
