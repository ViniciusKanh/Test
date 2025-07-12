import itertools
from collections import Counter
from marca.data_structures import CARs
import numpy as np
from sklearn.base import BaseEstimator
from marca.extract._extract import Extract
from marca.extract._msapriori import MSApriori


class MSAprioriPartitioned(MSApriori, BaseEstimator, Extract):
    def __init__(
            self,
            max_len=15,
            max_sup_diff=1,
            max_num_rules=20_000,
            B=0.25,
            name="",
            interest_measures=None,
    ):
        super().__init__(max_len=max_len, max_sup_diff=max_sup_diff, max_num_rules=max_num_rules, B=B, interest_measures=interest_measures)
        self.name = "MSAprioriPartitioned" + name

    def __call__(self, X, y):
        final_cars = CARs([])
        self.transform_to_sequential(X, y)
        self.data_transformed_original = self.data_transformed.copy()

        # Partition the data
        for y_p in np.unique(y):
            self.data_transformed = self.data_transformed_original[y == y_p]
            self.number_instances = len(self.data_transformed)
            rules_p = self.extract_rules()
            self.recalculate_interest_measures(rules_p, X.astype(float), X[y != y_p].astype(float))
            final_cars.extend(rules_p)

        return final_cars

    def recalculate_interest_measures(self, cars: CARs, x, x_not_p):
        length_x_p = len(x) - len(x_not_p)

        for car in cars:
            freq_not_covered_in_x_p = len(car.match_A(x_not_p))
            freq_covered_in_x_p = length_x_p*car.sup_antecedent
            car.set_sup_antecedent((freq_covered_in_x_p + freq_not_covered_in_x_p) / len(x))
            car.set_sup_consequent(car.sup_consequent * length_x_p / len(x))
            car.set_support(car.support * length_x_p / len(x))
            car.set_N(len(x))
            car.confidence = car.support / car.sup_antecedent

        cars.set_measures(cars.A, cars.B, cars.AB, cars.get_n_transactions()[0], self.interest_measures)

    def extract_rules(self):
        self.cars = CARs()
        self.set_must_have()
        self.set_MIS()
        self.set_support()

        self.level1_candidate_gen()
        self.level2_candidate_gen()
        self.generate_rules(level=2)

        level = 3
        while (self.candidates[level - 1]) and (level <= self.max_len) and (len(self.cars) < self.max_num_rules):
            self.leveln_candidate_gen(level)
            self.generate_rules(level=level)
            level += 1

        return self.cars
