import numpy as np
from ._rank import Rank


def similarity(interest_measures, method="manhattan"):
    reference_values = interest_measures.max(axis=0)

    if method == "manhattan":
        distance = (
                abs(reference_values-interest_measures).sum(axis=1)
                /reference_values.shape[0]
        )

    elif method == "euclidian":
        distance = np.sqrt(
            ((abs(reference_values-interest_measures) ** 2).sum(axis=1))
        )

    else:
        raise Exception(
            "Select manhattan or euclidian distance for method to measure distance."
        )

    return -distance


class SimilarityRank(Rank):
    def __init__(self, method="manhattan", measures=None, name=""):
        super().__init__()
        self.name = "Similarity"+name
        self.method = method
        self.measures = measures

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(measures=measures, normalized="rank")

        rank_values = similarity(interest_measures_values, self.method)
        return rank_values


if __name__ == "__main__":
    by = similarity(np.array([[0.2, 0.67, 0.02], [0.1, 0.5, 0], [0.2, 0.33, 0.10]]))

    r = np.array(
        sorted(
            enumerate(["R1", "R02", "R003"]),
            key=lambda x: (by[x[0]], len(x)),
            reverse=True,
        )
    )[:, 1].tolist()

    print(by)
    print(r)
