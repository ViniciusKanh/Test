import fim
import pandas as pd
from marca.data_structures import CAR, CARs
import numpy as np
from ._extract import Extract


class Apriori(Extract):
    def __init__(
            self,
            support=0.1,
            confidence=0.5,
            max_len=5,
            name="",
            interest_measures=None,
            remove_redundant=False,
    ):
        super().__init__()
        self.name = "Apriori" + name

        self.support = support
        self.confidence = confidence
        self.max_len = max_len
        self.interest_measures = interest_measures
        self.remove_redundant = remove_redundant

        self.rules = None

    def _create_cars(self, X, rules):
        cars = CARs()
        frequencies = []

        for rule in rules:
            _, _, f_lhs_rhs, _, f_lhs, f_rhs = rule

            car = CAR.from_fim(itemset=rule, support_rule=f_lhs_rhs, sup_antecedent=f_lhs, sup_consequent=f_rhs,
                               n_transactions=X.shape[1])

            frequencies.append([f_lhs_rhs, f_lhs, f_rhs])
            cars.append(car)

        self._set_interest_measures(cars, frequencies, X.shape[1])

        return cars

    def _check_redundant(self, rules):
        if self.remove_redundant:
            aux = pd.DataFrame(rules)
            aux = aux.drop_duplicates(subset=[1], keep=False)
            rules = aux.values.tolist()
            del aux
        return rules

    def _set_interest_measures(self, cars, frequencies, n):
        frequencies = np.array(frequencies)
        f_lhs_rhs = frequencies[:, 0].reshape(-1, 1)
        f_lhs = frequencies[:, 1].reshape(-1, 1)
        f_rhs = frequencies[:, 2].reshape(-1, 1)

        cars.set_measures(sup_antecedent=f_lhs, sup_consequent=f_rhs, sup_rule=f_lhs_rhs, num_transactions=n,
                          interest_measures=self.interest_measures)

    def _format_data(self, X, y):
        data = np.hstack((X, y.reshape(-1, 1)))
        string_rep = [
            [
                str(column) + ":=:" + data[line, column]
                for column in range(data.shape[1])
            ]
            for line in range(data.shape[0])
        ]

        appear = {None: "a"}
        appear.update({str(data.shape[1] - 1) + ":=:" + y_: "c" for y_ in np.unique(y)})
        return string_rep, appear

    def __call__(self, X, y):
        self._validate_data(X, y)
        X, y = self._transform_data(X, y)

        data, appear = self._format_data(X, y)

        rules = fim.apriori(
            data,
            supp=self.support * 100,
            conf=self.confidence * 100,
            mode="o",
            target="r",
            report="scxy",
            appear=appear,
            zmin=2,
            zmax=self.max_len,
        )
        self.non_formated_rules = rules

        if len(rules) == 0:
            raise Exception("Zero rules extract")

        rules = self._check_redundant(rules)
        self.rules = self._create_cars(X, rules)
        return self.rules
