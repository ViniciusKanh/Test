from ._rank import Rank
import numpy as np
import scipy.stats as ss


class Borda:
    def __init__(self, method):
        self.method = method

        if self.method == "mean":
            self.sort_function = self.borda_mean

        if self.method == "median":
            self.sort_function = self.borda_median

        if self.method == "l2":
            self.sort_function = self.borda_l2

        if self.method == "geometric":
            self.sort_function = self.borda_geometric

    def borda_mean(self, rules):
        return rules.mean(axis=1)

    def borda_median(self, rules):
        return np.median(rules, axis=1)

    def borda_geometric(self, rules):
        # print(rules)
        return ss.mstats.gmean(rules, axis=1)

    def borda_l2(self, rules):
        return ((rules**2).mean(axis=1)) ** (1 / len(rules))

    def fit(self, values):
        values_rank = ss.rankdata(values, axis=0, method="average")

        return self.sort_function(values_rank)

class BordaRank(Rank):
    def __init__(self, method, measures=None, name=""):
        """
        Parameters
        ----------
        method: str
            Method to be used in the Borda count. Options: "mean", "median", "l2", "geometric"

        measures: InterestMeasuresGroup
            Interest measures to be returned

        name: str
            Name of the rank method
        """
        super().__init__()
        self.name = "Borda" + method.capitalize() + name
        self.method = method
        self.measures = measures

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        mos = rules.get_measures(measures=measures, normalized="rank")
        rank_values = Borda(self.method).fit(mos)
        return rank_values


if __name__ == "__main__":
    by = Borda("l2").fit(np.array([[2.5, 3.0, 2.0], [1.0, 2.0, 1.0], [2.5, 1.0, 3.0]]))

    r = np.array(
        sorted(
            enumerate(["R1", "R02", "R00003"]),
            key=lambda x: (by[x[0]], len(x)),
            reverse=True,
        )
    )[:, 1].tolist()

    print(by)
    print(r)
