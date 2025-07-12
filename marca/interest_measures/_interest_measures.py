import numpy as np
import pandas as pd
from ._equations import InterestMeasuresEquations


class InterestMeasures(InterestMeasuresEquations):
    measures_available = {
        "accuracy": "ACC",
        "added_value": "AV",
        "chi_square": "Q2",
        "collective_strength": "CS",
        "complement_class_support": "CCS",
        "conditional_entropy": "CE",
        "confidence": "CON",
        "confidence_causal": "CDC",
        "confirm_causal": "CRC",
        "confirm_descriptive": "CRD",
        "confirmed_confidence_causal": "CCC",
        "correlation_coefficient": "CCO",
        "cosine": "COS",
        "coverage": "COV",
        "dir": "DIR",
        "f_measure": "FM",
        "gini_index": "GI",
        "goodman_kruskal": "GK",
        "implication_index": "IIN",
        "j_measure": "JM",
        "kappa": "k",
        "klosgen": "KLO",
        "k_measure": "KM",
        "kulczynski_2": "KU2",
        "least_contradiction": "LEC",
        "leverage": "LEV",
        "lift": "LIF",
        "loevinger": "LOE",
        "logical_necessity": "LON",
        "mutual_information": "MI",
        "normalized_mutual_information": "NMI",
        "odd_multiplier": "OM",
        "odds_ratio": "OR",
        "one_way_support": "1WS",
        "piatetsky_shapiro": "PS",
        "prevalence": "PRE",
        "putative_causal_dependency": "PCD",
        "recall": "REC",
        "relative_risk": "REL",
        "specificity": "SPE",
        "support": "SUP",
        "theil_uncertainty_coefficiente": "TUC",
        "tic": "TIC",
        "two_way_support": "2WS",
        "modified_lift": "MLIF",
        "dm2": "DM2",
        "dm3": "DM3",
        "dm4": "DM4",

    }

    def __init__(
        self,
        A,
        B,
        AB,
        N=10,
        measures=None,
    ):
        """
        Initializes an Interest Measures.

        :param A: A list representing the probabilities for events in A. List or numpy array.
        :param B: A list representing the probabilities for events in B. List or numpy array.
        :param AB: A list representing the probabilities for events in AB. List or numpy array.
        :param N: An integer representing the number of transactions. Default is 10.
        :param measures: Optional. A list of measures. Default is None.
        """

        super().__init__()

        self.AB = np.array(AB).reshape(-1, 1)
        self.A = np.array(A).reshape(-1, 1)
        self.B = np.array(B).reshape(-1, 1)
        self.N = N
        self.chosen_measures = measures

        self._set_probability()
        self._set_measures()

    def _set_probability(self):
        self.notB = 1 - self.B  # P(~B)
        self.notA = 1 - self.A  # P(~self.A)

        self.AnotB = self.A - self.AB  # P(self.A~B)
        self.notAB = self.B - self.AB  # P(~self.AB)
        self.notAnotB = 1 - self.B - self.A + self.AB  # P(~self.A~B)

        self.B_A = div_to_prob(self.AB, self.A)  # P(B|self.A)
        self.A_B = div_to_prob(self.AB, self.B)  # P(self.A|B)

        self.B_notA = div_to_prob(self.notAB, self.notA)  # P(B|~self.A)
        self.A_notB = div_to_prob(self.AnotB, self.notB)  # P(self.A|~B)

        self.notB_notA = 1 - self.B_notA  # P(~B|~self.A)
        self.notA_notB = 1 - self.A_notB  # P(~self.A|~B)

        self.notB_A = 1 - self.B_A  # P(~B|self.A)
        self.notA_B = 1 - self.A_B  # P(~self.A|B)

        self._round_measures()

    def _round_measures(self):
        round_value = 10

        # Round to 10 decimal places all initial probabilities
        self.AB = np.round(self.AB, round_value)
        self.A = np.round(self.A, round_value)
        self.B = np.round(self.B, round_value)
        self.notB = np.round(self.notB, round_value)
        self.notA = np.round(self.notA, round_value)
        self.AnotB = np.round(self.AnotB, round_value)
        self.notAB = np.round(self.notAB, round_value)
        self.notAnotB = np.round(self.notAnotB, round_value)
        self.B_A = np.round(self.B_A, round_value)
        self.A_B = np.round(self.A_B, round_value)
        self.B_notA = np.round(self.B_notA, round_value)
        self.A_notB = np.round(self.A_notB, round_value)
        self.notB_notA = np.round(self.notB_notA, round_value)
        self.notA_notB = np.round(self.notA_notB, round_value)
        self.notB_A = np.round(self.notB_A, round_value)
        self.notA_B = np.round(self.notA_B, round_value)

    def set_measures(self, measures):
        self.chosen_measures = measures
        self._set_measures()

    def _set_measures(self):
        if self.chosen_measures is None:
            self.chosen_measures = list(self.measures_available.keys())
        for measure in self.chosen_measures:
            setattr(self, measure, getattr(self, "_" + measure)())

    def get_chosen_measures(self):
        return self.chosen_measures

    def get_measure(self, measure):
        try:
            return getattr(self, measure)

        except:
            return getattr(self, "_" + measure)()

    def get_measures(self, measures=None, abbreviation=False):
        if measures is None:
            measures = self.chosen_measures

        if not abbreviation:
            return {measure: self.get_measure(measure) for measure in measures}

        return {
            self.measures_available[measure]: self.get_measure(measure)
            for measure in measures
        }

    def _repr_html_(self):
        df = pd.DataFrame({measure: self.get_measure(measure).reshape(-1) for measure in self.chosen_measures})
        return df._repr_html_()

    def as_df(self, abbreviation=False):
        return pd.DataFrame({measure: self.get_measure(measure).reshape(-1) for measure in self.chosen_measures})

    def get_measures_values(self):
        return np.array([self.get_measure(measure) for measure in self.chosen_measures]).T[0]

    def as_records(self):
        """
        Return list of dicts with measures and values.
        :return:
        """
        return [
            {measure: self.get_measure(measure)} for measure in self.chosen_measures
        ]

    def to_list(self):
        """
        Return list of dicts with measures and values.
        :return:
        """
        measures_values = self.get_measures_values()

        return [
            dict(zip(self.chosen_measures, measures_values[i]))
            for i in range(len(self.A))
        ]

    def __len__(self):
        return len(self.A)


def div_to_prob(numerator, denominator):
    return np.where(denominator == 0, 1, numerator / denominator)


if __name__ == "__main__":
    aux = InterestMeasures(
        measures=[
            "confidence",
            "lift",
            "complement_class_support",
            "j_measure",
            "zhang",
        ]
    )
    print(aux.as_df())
