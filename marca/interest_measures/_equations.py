import numpy as np

np.seterr(divide="ignore", invalid="ignore")


def div_check_zero(numerator, denominator):
    return np.where(denominator == 0, np.inf, numerator / denominator)


def positive(function):
    def wrapper(self):
        im_values = function(self)
        return im_values

    return wrapper


def negative(function):
    def wrapper(self):
        im_values = function(self)
        return -1 * im_values

    return wrapper


class InterestMeasuresEquations:
    def __init__(self):
        self.A = None
        self.B = None
        self.AB = None
        self.notA = None
        self.notB = None
        self.AnotB = None
        self.notAB = None
        self.notAnotB = None
        self.B_A = None
        self.A_B = None
        self.B_notA = None
        self.A_notB = None
        self.notB_notA = None
        self.notA_notB = None
        self.notB_A = None
        self.notA_B = None
        self.N = None

    @positive
    def _one_way_support(self):
        """
        Calculate the one-way support of the rule.

        Equation
        --------

        1WS = P(B|A)*log2(P(B|A)/P(B))

        References
        ----------

        [1] https://link.springer.com/article/10.1007/s10618-013-0326-x

        """
        return self.B_A * np.log2(self.B_A / self.B)

    @positive
    def _two_way_support(self):
        """
        Calculate the two-way support of the rule.

        Equation
        --------

        2WS = P(AB)*log2(P(B|A)/P(B))

        References
        ----------

        [1] https://link.springer.com/article/10.1007/s10618-013-0326-x


        Returns
        -------

        """
        return self.AB * np.log2(self.AB / (self.A * self.B))

    @positive
    def _accuracy(self):
        return self.AB + self.notAnotB

    @positive
    def _added_value(self):
        return self.B_A - self.B

    @positive
    def _chi_square(self):
        part1 = ((self.AB - (self.A * self.B)) ** 2) * self.N
        part2 = self.A * self.notA * self.B * self.notB

        return div_check_zero(part1, part2)

    @positive
    def _collective_strength(self):
        part1 = (self.AB + self.notAnotB) / (
                (self.A * self.B) + (self.notA * self.notB)
        )
        part2_1 = 1 - (self.A * self.B) - (self.notA * self.notB)
        part2_2 = np.around((1 - self.AB - self.notAnotB), 10)

        part2 = div_check_zero(part2_1, part2_2)

        return part1 * part2

    @negative
    def _complement_class_support(self):
        part1 = self.AnotB
        part2 = self.notB

        return div_check_zero(part1, part2)

    @positive
    def _conditional_entropy(self):
        part1 = -1 * self.B_A * np.log2(self.B_A)
        part2 = np.where(self.notB_A == 0, 0, -1 * self.notB_A * np.log2(self.notB_A))

        return part1 + part2

    @negative
    def _conditional_entropy_neg(self):
        part1 = -1 * self.B_A * np.log2(self.B_A)
        part2 = np.where(self.notB_A == 0, 0, -1 * self.notB_A * np.log2(self.notB_A))

        return part1 + part2

    # Boa alternativa
    @positive
    def _conditional_entropy_alt(self):
        part1 = -1 * self.B_A ** 8 * np.log2(self.B_A)
        part2 = np.where(self.notB_A == 0, 0, -1 * self.notB_A ** (16) * np.log2(self.notB_A))

        return part1 + part2

    @positive
    def _confidence(self):
        return self.B_A

    @positive
    def _confidence_causal(self):
        return (self.B_A + self.notA_notB) / 2

    @positive
    def _confirm_causal(self):
        return self.AB + self.notAnotB - (2 * self.AnotB)

    @positive
    def _confirm_descriptive(self):
        return self.AB - self.AnotB

    @positive
    def _confirmed_confidence_causal(self):
        return ((self.B_A + self.notA_notB) / 2) - self.notB_A

    @positive
    def _conviction(self):
        part1 = self.A * self.notB
        part2 = self.AnotB

        return div_check_zero(part1, part2)

    @positive
    def _correlation_coefficient(self):
        part1 = self.AB - (self.A * self.B)
        part2 = np.sqrt(self.A * self.B * self.notA * self.notB)

        return div_check_zero(part1, part2)

    @positive
    def _cosine(self):
        part1 = self.AB
        part2 = np.sqrt(self.A * self.B)

        return part1 / part2

    @positive
    def _coverage(self):
        return self.A

    @positive
    def _dir(self):
        result = np.zeros(self.B.shape)

        result = np.where(np.logical_and((self.B <= 0.5), (self.B_A <= 0.5)), 0, result)

        result = np.where(
            np.logical_and(
                np.logical_and((self.B <= 0.5), (self.B_A > 0.5)), (self.B_A != 1)
            ),
            1 + (self.B_A * np.log2(self.B_A)) + (self.notB_A * np.log2(self.notB_A)),
            result,
        )

        result = np.where(
            np.logical_and(
                np.logical_and((self.B <= 0.5), (self.B_A > 0.5)), (self.B_A == 1)
            ),
            1,
            result,
        )

        result = np.where(
            np.logical_and((self.B > 0.5), (self.B_A <= 0.5)),
            1 + (1 / (self.B * np.log2(self.B) + self.notB * np.log2(self.notB))),
            result,
        )

        result = np.where(
            np.logical_and(
                np.logical_and((self.B > 0.5), (self.B_A > 0.5)), (self.B_A != 1)
            ),
            1 - ((self.B_A * np.log2(self.B_A) + self.notB_A * np.log2(self.notB_A)) / (
                    self.B * np.log2(self.B) + self.notB * np.log2(self.notB))),
            result,
        )

        result = np.where(
            np.logical_and(np.logical_and((self.B > 0.5), (self.B_A > 0.5)), (self.B_A == 1)),
            1 - ((self.B_A * np.log2(self.B_A)) / (self.B * np.log2(self.B) + self.notB * np.log2(self.notB))),
            result,
        )

        result = np.where(self.B == 1, -np.inf, result)

        return result

    def _dir_for_tic(self, A, B, AB):
        B_A = np.around(AB / A, 11)
        notB = 1 - B
        notB_A = np.around(1 - B_A, 11)

        result = np.zeros(B.shape)

        result = np.where(np.logical_and((B <= 0.5), (B_A <= 0.5)), 0, result)

        result = np.where(
            np.logical_and(np.logical_and((B <= 0.5), (B_A > 0.5)), (B_A != 1)),
            1 + B_A * np.log2(B_A) + notB_A * np.log2(notB_A),
            result,
        )

        result = np.where(
            np.logical_and(np.logical_and((B <= 0.5), (B_A > 0.5)), (B_A == 1)),
            1 + B_A * np.log2(B_A),
            result,
        )

        result = np.where(
            np.logical_and((B > 0.5), (B_A <= 0.5)),
            1 + (1 / (B * np.log2(B) + notB * np.log2(notB))),
            result,
        )

        result = np.where(
            np.logical_and(np.logical_and((B > 0.5), (B_A > 0.5)), (B_A != 1)),
            1
            - (B_A * np.log2(B_A) + notB_A * np.log2(notB_A))
            / (B * np.log2(B) + notB * np.log2(notB)),
            result,
        )

        result = np.where(
            np.logical_and(np.logical_and((B > 0.5), (B_A > 0.5)), (B_A == 1)),
            1 - (B_A * np.log2(B_A)) / (B * np.log2(B) + notB * np.log2(notB)),
            result,
        )

        result = np.where(B == 1, -np.inf, result)

        return result

    @positive
    def _exemple_and_counterexemple_rate(self):
        return 1 - (self.AnotB / self.AB)

    @positive
    def _f_measure(self):
        part1 = 2 * self.A_B * self.B_A
        part2 = self.A_B + self.B_A

        return part1 / part2

    @positive
    def _ganascia(self):
        return (2 * self.B_A) - 1

    @positive
    def _gini_index(self):
        part1 = self.A * ((self.B_A ** 2) + (self.notB_A ** 2))
        part2 = self.notA * ((self.B_notA ** 2) + (self.notB_notA ** 2))
        part3 = self.B ** 2
        part4 = self.notB ** 2

        return part1 + part2 - part3 - part4

    @positive
    def _goodman_kruskal(self):
        part1 = (
                np.max([self.AB, self.AnotB], axis=0)
                + np.max([self.notAB, self.notAnotB], axis=0)
                + np.max([self.AB, self.notAB], axis=0)
                + np.max([self.AnotB, self.notAnotB], axis=0)
                - np.max([self.A, self.notA], axis=0)
                - np.max([self.B, self.notB], axis=0)
        )

        part2 = (
                2
                - np.max([self.A, self.notA], axis=0)
                - np.max([self.B, self.notB], axis=0)
        )

        return div_check_zero(part1, part2)

    @negative
    def _implication_index(self):
        part1 = self.AnotB - (self.A * self.notB)
        part2 = np.sqrt(self.A * self.notB)

        return div_check_zero(part1, part2)

    @positive
    def _information_gain(self):
        return np.log2((self.AB / (self.A * self.B)))

    @positive
    def _jaccard(self):
        return self.AB / (self.A + self.B - self.AB)

    @positive
    def _j_measure(self):
        part1 = self.AB * np.log2(self.B_A / self.B)
        part2 = np.where(
            (self.notB * self.notB_A) == 0,
            0,
            self.AnotB * np.log2(self.notB_A / self.notB),
        )

        return part1 + part2

    @positive
    def _kappa(self):
        part1 = (
                (self.B_A * self.A)
                + (self.notB_notA * self.notA)
                - (self.A * self.B)
                - (self.notA * self.notB)
        )
        part2 = 1 - (self.A * self.B) - (self.notA * self.notB)

        return div_check_zero(part1, part2)

    @positive
    def _klosgen(self):
        return np.sqrt(self.A) * (self.B_A - self.B)

    @positive
    def _k_measure(self):
        part1 = self.B_A * np.log2(self.B_A / self.B)
        part3 = self.B_A * np.log2(self.B_A / self.notB)
        part2 = np.where(
            self.notB_notA == 0,
            0,
            (self.notB_notA * np.log2(self.notB_notA / self.notB)),
        )
        part4 = np.where(
            self.notB_notA == 0, 0, (self.notB_notA * np.log2(self.notB_notA / self.B))
        )

        return part1 + part2 - part3 - part4

    @positive
    def _kulczynski_1(self):
        part1 = self.AB
        part2 = self.AnotB + self.notAB

        return div_check_zero(part1, part2)

    @positive
    def _kulczynski_2(self):
        return ((self.AB / self.A) + (self.AB / self.B)) / 2

    @positive
    def _laplace_correction(self):
        return (self.N * self.AB + 1) / (self.N * self.A + 2)

    @positive
    def _least_contradiction(self):
        return (self.AB - self.AnotB) / self.B

    @positive
    def _leverage(self):
        return self.B_A - (self.A * self.B)

    @positive
    def _lift(self):
        return self.AB / (self.A * self.B)

    @positive
    def _loevinger(self):
        return np.where(self.notB == 0, 0, 1 - (self.AnotB / (self.A * self.notB)))

    @negative
    def _logical_necessity(self):
        part1 = self.notA_B
        part2 = self.notA_notB

        return div_check_zero(part1, part2)

    @positive
    def _mutual_information(self):
        part1 = self.AB * np.log2(self.AB / (self.A * self.B))
        part2 = np.where(
            self.AnotB == 0, 0, self.AnotB * np.log2(self.AnotB / (self.A * self.notB))
        )
        part3 = np.where(
            self.notAB == 0, 0, self.notAB * np.log2(self.notAB / (self.notA * self.B))
        )
        part4 = np.where(
            self.notAnotB == 0,
            0,
            self.notAnotB * np.log2(self.notAnotB / (self.notA * self.notB)),
        )

        return part1 + part2 + part3 + part4

    @positive
    def _normalized_mutual_information(self):
        part1 = np.around(self._mutual_information(), 10)
        part2 = np.where(
            self.A == 1,
            (-self.A * np.log2(self.A)),
            (-self.A * np.log2(self.A)) - (self.notA * np.log2(self.notA)),
        )

        return div_check_zero(part1, part2)

    @positive
    def _odd_multiplier(self):
        part1 = self.AB * self.notB
        part2 = self.B * self.AnotB

        return div_check_zero(part1, part2)

    @positive
    def _odds_ratio(self):
        part1 = self.AB * self.notAnotB
        part2 = self.AnotB * self.notAB

        return div_check_zero(part1, part2)

    @positive
    def _piatetsky_shapiro(self):
        return self.AB - (self.A * self.B)

    @positive
    def _prevalence(self):
        return self.B

    @positive
    def _putative_causal_dependency(self):
        part1 = (self.B_A - self.B) / 2
        part2 = self.notA_notB - self.notA
        part3 = self.notB_A - self.notB
        part4 = self.A_notB - self.A

        return part1 + part2 - part3 - part4

    @positive
    def _recall(self):
        return self.A_B

    @positive
    def _relative_risk(self):
        part1 = self.B_A
        part2 = self.B_notA

        return div_check_zero(part1, part2)

    @positive
    def _sebag_schoenaure(self):
        part1 = self.AB
        part2 = self.AnotB

        return div_check_zero(part1, part2)

    @positive
    def _specificity(self):
        return self.notB_notA

    @positive
    def _support(self):
        return self.AB

    @positive
    def _theil_uncertainty_coefficiente(self):
        part1 = np.around(self._mutual_information(), 10)
        part2 = np.where(
            self.B == 1,
            (-self.B * np.log2(self.B)),
            (-self.B * np.log2(self.B)) - (self.notB * np.log2(self.notB)),
        )

        return div_check_zero(part1, part2)

    @positive
    def _tic(self):
        part2 = np.around(
            self._dir_for_tic(A=self.notB, B=self.notA, AB=self.notAnotB), 10
        )
        values = np.where(
            self.B == 1, -np.inf, np.sqrt(np.around(self._dir(), 10) * part2)
        )

        return values

    @positive
    def _yuleQ(self):
        part1 = (self.AB * self.notAnotB) - (self.AnotB * self.notAB)
        part2 = (self.AB * self.notAnotB) + (self.AnotB * self.notAB)

        return div_check_zero(part1, part2)

    @positive
    def _yuleY(self):
        part1 = np.sqrt(self.AB * self.notAnotB) - np.sqrt(self.AnotB * self.notAB)
        part2 = np.sqrt(self.AB * self.notAnotB) + np.sqrt(self.AnotB * self.notAB)

        return div_check_zero(part1, part2)

    @positive
    def _zhang(self):
        part1 = self.AB - (self.A * self.B)
        part2 = np.max([self.AB * (1 - self.B), self.B * (self.A - self.AB)], axis=0)

        return div_check_zero(part1, part2)

    @positive
    def _modified_lift(self):
        return self.notAnotB / self.AnotB

    @positive
    def _dm2(self):
        return self.notAnotB / (self.AnotB * np.sqrt(self.B))

    @positive
    def _dm3(self):
        return (self.notAnotB * self.A) / (self.AnotB * np.sqrt(self.B))

    @positive
    def _dm4(self):
        return self.notAnotB / (self.AnotB * np.sqrt(self.A))
