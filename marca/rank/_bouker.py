import numpy as np
from ._rank import Rank


class RankRule:
    def strictly_dominates_one_measure(self, r1, r2):
        return (r1 > r2).any()

    def dominates(self, r1, r2):
        return (r1 >= r2).all()

    def strictly_dominates(self, r1, r2):
        return self.dominates(r1, r2) and self.strictly_dominates_one_measure(r1, r2)

    def sky_rule(self, interest_measures):
        reference_values = interest_measures.max(axis=0)
        deg_sim = (
            abs(reference_values - interest_measures).sum(axis=1)
            / reference_values.shape[0]
        )

        sky = np.array([], dtype=int)
        C = np.arange(0, interest_measures.shape[0], dtype=int)
        E = np.arange(0, interest_measures.shape[0], dtype=int)
        D = np.argsort(deg_sim, kind="stable")

        # print(D)

        while C.shape[0] > 0:
            r_star, D = D[0], np.delete(D, 0)
            C = np.delete(C, np.where(C == r_star)[0][0])

            sky = np.append(sky, r_star)
            sr = []

            for e in E:
                if self.strictly_dominates(
                    interest_measures[r_star], interest_measures[e]
                ):
                    D = np.delete(D, np.where(D == e))
                    C = np.delete(C, np.where(C == e))

                elif self.strictly_dominates_one_measure(
                    interest_measures[e], interest_measures[r_star]
                ):
                    sr.append(e)

            E = sr

        return sky

    def fit(self, mos, flat_rank=False, rank_index=True):
        rank = []
        R = np.arange(0, mos.shape[0], dtype=int)

        while len(R) > 0:
            undominated_rules = self.sky_rule(mos[R])
            rank.append((R[undominated_rules]).tolist())
            R = np.delete(R, undominated_rules)

        if flat_rank:
            if rank_index:
                return -np.argsort([rule for Ep in rank for rule in Ep])

            else:
                return [rule for Ep in rank for rule in Ep]

        else:
            return rank


class SkylineRank(Rank):
    def __init__(self, method="td", measures=None, name=""):
        super().__init__()
        self.method = method
        self.name = "BoukerTD" + name
        self.measures = measures

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(
            measures=measures, normalized="rank"
        )
        rank_values = RankRule().fit(
            interest_measures_values, flat_rank=True, rank_index=True
        )

        return rank_values


if __name__ == "__main__":
    by = RankRule().fit(
        np.array([[0.2, 0.67, 0.02], [0.1, 0.5, 0], [0.2, 0.33, 0.10]]),
        flat_rank=True,
        rank_index=True,
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
