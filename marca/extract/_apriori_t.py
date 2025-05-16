from collections import Counter

import fim
import pandas as pd
from marca.interest_measures import InterestMeasures
from marca.data_structures import CAR, CARs
import numpy as np
from sklearn.base import BaseEstimator
from ._extract import Extract
from ._apriori import Apriori


class AprioriT(Apriori, BaseEstimator, Extract):
    def __init__(
            self,
            confidence,
            max_len,
            B=0.25,
            name="",
            interest_measures=None,
            # remove_redundant=False,
    ):
        """
        Class to extract Class Association
        Parameters
        ----------
        confidence
        max_len
        B
        name
        interest_measures
        """
        super().__init__(support=0, confidence=0, max_len=max_len, interest_measures=interest_measures)
        self.name = "AprioriT" + name
        self.confidence = confidence
        self.B = B
        self.rules = None

    def __call__(self, X, y):
        data, appear = self._format_data(X, y)

        # Apriori T set support
        self.support = self.B * (Counter(y).most_common()[-1][1] / len(y))

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

        if len(rules) == 0:
            raise Exception("Zero rules extract")

        rules = self._check_redundant(rules)
        self.rules = self._create_cars(X, rules)
        return self.rules
