import collections
import numpy as np
import pandas as pd
from marca.interest_measures._interest_measures import InterestMeasures

class CAR:
    id = 0

    def __init__(self, antecedent, consequent, support, sup_antecedent, sup_consequent, n_transactions):
        """
        Class to represent a Class Association Rule
        Parameters
        ----------
        antecedent
        consequent
        support
        sup_antecedent
        sup_consequent
        n_transactions
        """

        # Id of the rule
        self.rid = CAR.id
        CAR.id += 1

        self.antecedent = antecedent
        self.consequent = consequent
        self.rule = np.hstack((self.antecedent, self.consequent))

        self.sup_antecedent = sup_antecedent
        self.sup_consequent = sup_consequent
        self.n_transactions = n_transactions

        self.support = support
        self.confidence = self.support / sup_antecedent

        self.empty_att = np.isnan(self.antecedent)
        self.empty_att_and_class = np.isnan(self.rule)
        self.length = len(antecedent[self.empty_att])+1
        self.rank_order = -np.inf
        self.interest_measures = collections.defaultdict(dict)

    def __repr__(self):
        # Show antecedent consequent and id of the rule
        return f"({self.rid}) {self.antecedent} -> {self.consequent} (Sup={self.support}, Conf= {self.confidence})"

    def set_A(self, A):
        self.sup_antecedent = A

    def set_B(self, B):
        self.sup_consequent = B

    def set_AB(self, AB):
        self.support = AB

    def set_N(self, N):
        self.n_transactions = N

    def set_sup_antecedent(self, sup_antecedent):
        self.sup_antecedent = sup_antecedent

    def set_sup_consequent(self, sup_consequent):
        self.sup_consequent = sup_consequent

    def set_support(self, support):
        self.support = support

    def set_n_transactions(self, n_transactions):
        self.n_transactions = n_transactions

    def set_interest_measures(self, interest_measures):
        self.interest_measures = interest_measures

    def set_measures(self, A, B, AB, N, interest_measures):
        self.set_A(A)
        self.set_B(B)
        self.set_AB(AB)
        self.set_N(N)
        self.set_interest_measures(interest_measures)

    def update_interest_measures(self, interest_measures):
        self.interest_measures.update(interest_measures)

    def get_measure(self, measure):
        return self.interest_measures[measure]


    def update_confidence(self):
        self.confidence = self.support / self.sup_antecedent

    @property
    def AB(self):
        return self.support

    @classmethod
    def from_fim(cls, itemset, support_rule, sup_antecedent, sup_consequent, n_transactions):
        items = np.array([item.split(":=:") for item in itemset[1]], dtype="float").T
        antecedent = np.full([n_transactions], np.nan, dtype="float")
        antecedent[items[0].astype(int)] = items[1]
        consequent = np.array(itemset[0].split(":=:")[1], dtype="float")

        return cls(antecedent, consequent, support_rule, sup_antecedent, sup_consequent, n_transactions)

    @classmethod
    def from_numpy(cls, rule, sup_rule, sup_antecedent, sup_consequent, n_transactions):
        antecedent = rule[:-1]
        consequent = rule[-1]

        return cls(antecedent, consequent, sup_rule, sup_antecedent, sup_consequent, n_transactions)

    def match_AB(self, data):
        """
        Return all the indexes of the data that are covered by the rule with the consequent
        Parameters
        ----------
        data

        Returns
        -------

        """
        index_covered = np.where((self.empty_att_and_class | (self.rule == data)).all(axis=1))[0]
        return index_covered

    # def match_A(self, data):
    #     index_covered = np.where((self.empty_att | (self.antecedent == data)).all(axis=1))[0]
    #     return index_covered

    def match_A(self, data):
        """
        Return all the indexes of the data that are covered by the rule without the consequent
        Parameters
        ----------
        data

        Returns
        -------

        """
        # Check if data is uni-dimensional
        if len(data.shape) == 1:
            return (self.antecedent[~self.empty_att] == data[~self.empty_att]).all()

        else:
            return np.where((self.antecedent[~self.empty_att] == data[:, ~self.empty_att]).all(axis=1))[0]

    def match(self, data):
        """
        Return if a instance is covered by the rule without the consequent
        Parameters
        ----------
        data

        Returns
        -------

        """
        return (self.empty_att | (self.antecedent == data)).all()

    def match_complete(self, data):
        """
        Return if a instance is covered by the rule without the consequent
        Parameters
        ----------
        data

        Returns
        -------

        """
        return (self.empty_att_and_class | (self.rule == data)).all()

    def _repr_html_(self):
        df_repr = pd.DataFrame(
            [
                {
                    "ID": self.rid,
                    "Antecedent": self.antecedent,
                    "Consequent": self.consequent,
                    "Support": self.support,
                    "Confidence": self.confidence,
                }
            ]
        )

        return df_repr._repr_html_()