from marca.rank._rank import Rank
import numpy as np


class Topsis:
    def __init__(self):
        pass

    def fit(self, data, weights):
        """
        Topsis algorithm implemented without criterion type (consider only max criterion)
        Parameters
        ----------
        data
        weights

        Returns
        -------

        """
        normalized_data = (data / (np.sqrt(np.sum((data**2), axis=0)))) * weights

        max_values = normalized_data.max(axis=0)
        min_values = normalized_data.min(axis=0)

        dist_top = np.sqrt(np.sum((normalized_data - max_values) ** 2, axis=1))
        dist_bottom = np.sqrt(np.sum((normalized_data - min_values) ** 2, axis=1))

        if ((dist_top + dist_bottom) == 0).all():
            return np.ones(shape=(len(data)))

        rank_value = dist_bottom / (dist_top + dist_bottom)

        return rank_value


class TopsisRank(Rank):
    def __init__(self, measures=None, name=""):
        super().__init__()
        self.name = "Topsis" + name
        self.measures = measures

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(
            measures=measures, normalized="rank"
        )
        weights = measures.get_weights()

        rank_values = Topsis().fit(interest_measures_values, weights)

        return rank_values


if __name__ == "__main__":
    by = Topsis().fit(
        np.array([[1, 1, 2, 4], [2, 2, 2, 5], [5, 0, 5, 6], [5, 0, 5, 6]]),
        np.array([1, 1, 4, 1]),
    )

    r = np.array(
        sorted(
            enumerate(["R1", "R02", "R003", "R0004"]),
            key=lambda x: (by[x[0]], len(x)),
            reverse=True,
        )
    )[:, 1].tolist()

    print(r)
