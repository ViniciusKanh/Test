from ._rank import Rank
from ..interest_measures._interest_measures_groups import InterestMeasuresGroup


class InterestMeasureRank(Rank):
    def __init__(self):
        super().__init__()
        self.name = 'IndividualRank'

    def set_measure(self, measure):
        # self.measure = InterestMeasuresGroup(name=measure, measures=measure)
        self.name = f"IM({measure.get_measures()[0].capitalize()})"

    # def __repr__(self):
    #     return f"RankByInterestMeasure({self.name.capitalize()})"

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        if len(measures.get_measures()) > 1:
            raise ValueError(
                "InterestMeasureRank only accepts one measure at a time"
            )

        rank_values = rules.get_measure(measures.get_measures()[0])
        return rank_values
