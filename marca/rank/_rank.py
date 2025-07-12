from ..data_structures import CARs
from ..interest_measures import InterestMeasuresGroup


class Rank:
    """
    Abstract class for ranking rules methods
    """

    def __init__(self):
        """
        Abstract class for ranking rules methods
        """
        self._rank_values = None
        self.measures = None

    def __repr__(self):
        return f"Rank({self.name}) - Interest Measure({self.measures})"

    def set_measures(self, measures):
        """

        Parameters
        ----------
        measures: InterestMeasuresGroup
                  InterestMeasuresGroup object with measures to be used

        Returns
        -------

        """
        self.measures = measures

    def __call__(self, x, y, rules, measures):
        """
        Parameters
        ----------
        x
        y
        rules: CARs
            CARs object with rules to be ranked

        measures: InterestMeasuresGroup or str
            InterestMeasuresGroup object with measures to be used

        Returns
        -------
        cars: CARs
            CARs object with ranked rules

        Examples
        --------
        >>> from marca.data_structures import CARs
        >>> from marca.interest_measures import InterestMeasuresGroup
        >>> from marca.extract import Apriori
        >>> from marca.rank import Rank
        >>> from marca.datasets import load_iris

        >>> X_train, y_train, X_test, y_test = load_iris(fold=0)
        >>> rules = Apriori(support=0.1, confidence=0.5, max_len=3).fit_transform(X_train, y_train)
        >>> measures = InterestMeasuresGroup(["support", "confidence", "lift"])
        >>> rank = Rank()
        >>> cars_ranked = rank.fit_transform(rules, measures)
        >>> cars_ranked

        """
        pass

    def rankmethod(function):
        def wrapper(self, x, y, rules, measures):
            if isinstance(measures, str):
                measures = InterestMeasuresGroup(name=measures, measures=[measures])

            if isinstance(measures, list):
                measures = InterestMeasuresGroup(name="DefaultGroupName", measures=measures)

            if isinstance(measures, tuple or set):
                measures = InterestMeasuresGroup(name="DefaultGroupName", measures=list(measures))

            self._rank_values = function(self, x, y, rules, measures)
            return rules.sorted(by=self._rank_values)

        return wrapper

    rankmethod = staticmethod(rankmethod)

    def get_rank_values(self):
        return self._rank_values
