import numpy as np
import scipy.stats as ss
from ._rank import Rank


class CBARank(Rank):
    def __init__(self):
        super().__init__()
        self.name = "CBARank"

    def __repr__(self):
        return (
            f"Rank(CBA) - Interest Measure([Confidence, Support, Rule Length, Rule ID])"
        )

    def sort_old(self, rules):
        rules.sorted_cba()
        print([r.rid for r in rules])

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):

        if measures is None:
            measures = ["support", "confidence"]
        by = [
            idx
            for idx, r in sorted(
                enumerate(rules),
                key=lambda x: (x[1].confidence, x[1].support, -x[1].length, -x[1].rid),
                reverse=True,
            )
        ]

        rank_values = ss.rankdata(-np.argsort(by)) / len(by)
        return rank_values

    def get_rank_values(self, rules, measures):
        by = [
            idx
            for idx, r in sorted(
                enumerate(rules),
                key=lambda x: (x[1].confidence, x[1].support, -x[1].length, -x[1].rid),
                reverse=True,
            )
        ]

        rank_values = ss.rankdata(-np.argsort(by)) / len(by)
        return rank_values
