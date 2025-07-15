from ._rank import Rank
from ..interest_measures._interest_measures_groups import InterestMeasuresGroup


class InterestMeasureRank(Rank):
    """Rank rules by a single interest measure."""

    def __init__(self, measure=None):
        super().__init__()
        self.name = 'IndividualRank'
        if measure is not None:
            self.set_measure(measure)

    def set_measure(self, measure):
        """Configure which interest measure this ranker should use."""
        if isinstance(measure, str):
            measure = InterestMeasuresGroup(name=measure, measures=[measure])
        self.name = f"IM({measure.get_measures()[0].capitalize()})"
        self._measure = measure

    # def __repr__(self):
    #     return f"RankByInterestMeasure({self.name.capitalize()})"

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        measure_name = measures.get_measures()[0]
        # If more than one measure is provided we simply use the first one as
        # done when ``set_measure`` receives a string.  This keeps the behaviour
        # simple for the unit tests in this repository.
        rank_values = rules.get_measure(measure_name)
        return rank_values
