from ._rank import Rank
import numpy as np


class WSMRank(Rank):
    def __init__(self, measures=None, name=""):
        super().__init__()
        self.name = "WSM" + name
        self.measures = measures

    def wsm(self, data, weights):
        return np.sum(data * weights, axis=1)

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(
            measures=measures, normalized="rank"
        )
        weights = measures.get_weights()

        rank_values = self.wsm(interest_measures_values, weights)

        return rank_values


class WPMRank(Rank):
    def __init__(self, measures=None, name=""):
        super().__init__()
        self.name = "WPM" + name
        self.measures = measures

    def wpm(self, data, weights):
        return np.prod(data**weights, axis=1)

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        interest_measures_values = rules.get_measures(
            measures=measures, normalized="rank"
        )
        weights = measures.get_weights()

        rank_values = self.wpm(interest_measures_values, weights)

        return rank_values
