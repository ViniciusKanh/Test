import collections
import pickle

import numpy as np
import pandas as pd
from collections import UserList
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
from marca.interest_measures import InterestMeasures, InterestMeasuresGroup
from ._car import CAR


class CARs(UserList):
    ####################################
    # Build-in functions ###############
    ####################################

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.__class__(self.data[i])

        elif isinstance(i, (list, np.ndarray)):
            return self.__class__([self.data[j] for j in i])

        else:
            return self.data[i]

    def __repr__(self):
        return self.to_pandas().__repr__()

    def _repr_html_(self):
        return self.to_pandas()._repr_html_()

    ####################################
    # Properties functions #############
    ####################################

    @property
    def A(self):
        return np.array([rule.sup_antecedent for rule in self])

    @property
    def B(self):
        return np.array([rule.sup_consequent for rule in self])

    @property
    def AB(self):
        return np.array([rule.AB for rule in self])

    @property
    def interest_measures(self):
        """
        Return a numpy array with the interest measures of each rule in records format
        Returns
        -------
        numpy array
            Interest measures of each rule in records format
        """
        return np.array([rule.interest_measures for rule in self])

    @property
    def support(self):
        return np.array([rule.support for rule in self])

    @property
    def confidence(self):
        return np.array([rule.confidence for rule in self])

    @property
    def rules(self):
        return self.to_numpy()

    @property
    def antecedent(self):
        """
        Return list of antecedents of the rules
        Returns
        -------

        """
        return np.array([rule.antecedent for rule in self])

    @property
    def consequent(self):
        """
        Return list of consequents of the rules
        Returns
        -------

        """
        return np.array([rule.consequent for rule in self])

    ####################################
    # CRUD functions ###################
    ####################################

    def delete(self, i):
        """
        Delete the rules in the index i from the CARs object and return a new CARs object
        Parameters
        ----------
        i: [int | list | numpy array]

        Returns
        -------
        CARs object without the rules in the index i
        """
        if isinstance(i, (list, np.ndarray)):
            return self.__class__(
                [self.data[j] for j in range(len(self.data)) if j not in i]
            )

        else:
            return self.data[i]

    def filter(self, condition):
        """
        Filter the rules that match the condition
        Parameters
        ----------
        condition

        Returns
        -------

        """
        return self.__class__([rule for rule in self if condition(rule)])

    def swap(self, idx1, idx2):
        """
        Swap the rules in the idx1 and idx2 positions
        Parameters
        ----------
        idx1
        idx2

        Returns
        -------

        """
        aux = self[idx1]
        self[idx1] = self[idx2]
        self[idx2] = aux

    def update_interest_measures(self, interest_measures=None):
        """
        Update the interest measures of the rules
        Returns
        -------

        """
        A = self.get_sup_antecedent()
        B = self.get_sup_consequent()
        AB = self.get_support()
        N = self.get_n_transactions()[0]
        interest_measures = InterestMeasures(A=A, B=B, AB=AB, N=N, measures=interest_measures)
        list_of_im = interest_measures.to_list()

        for rule, measure in zip(self, list_of_im):
            rule.update_interest_measures(measure)

    def update_confidence(self):
        """
        Update the confidence of the rules
        Returns
        -------

        """
        for rule in self:
            rule.update_confidence()

    ####################################
    # Setters functions ################
    ####################################

    def set_sup_antecedent(self, A):
        """
        Set the A values of the rules
        Parameters
        ----------
        A

        Returns
        -------

        """
        for rule, a in zip(self, A):
            rule.sup_antecedent = a

    def set_sup_consequent(self, B):
        """
        Set the B values of the rules
        Parameters
        ----------
        B

        Returns
        -------

        """
        for rule, b in zip(self, B):
            rule.sup_consequent = b

    def set_support(self, AB):
        """
        Set the AB values of the rules
        Parameters
        ----------
        AB

        Returns
        -------

        """
        for rule, ab in zip(self, AB):
            rule.support = ab

    def set_measures(self, sup_antecedent, sup_consequent, sup_rule, num_transactions, interest_measures):
        """
        Set the interest measures of each rule
        Parameters
        ----------
        sup_antecedent: numpy array
        sup_consequent: numpy array
        sup_rule: numpy array
        num_transactions: [int | list | numpy array]
        interest_measures

        Returns
        -------
        """
        interest_measures = InterestMeasures(
            A=sup_antecedent, B=sup_consequent, AB=sup_rule, N=num_transactions, measures=interest_measures
        )
        list_of_im = interest_measures.to_list()

        for rule, sup_antecedent, sup_consequent, sup_rule, measure in zip(self, sup_antecedent, sup_consequent, sup_rule, list_of_im):
            rule.set_measures(sup_antecedent, sup_consequent, sup_rule, num_transactions, measure)

    def sorted(self, by):
        """
        Sort the rules by the interest measure passed as parameter
        Parameters
        ----------
        by: list or numpy array

        Returns
        -------
        CARs
            Sorted rules
        """
        by_minmax = self.__scaler_by(by)

        for r, rank_order in zip(self, by_minmax):
            r.rank_order = rank_order[0]

        return CARs(sorted(self, key=lambda x: (x.rank_order, x.support, -x.length, -x.rid), reverse=True))

    def __scaler_by(self, by):
        by_reshape = np.array(by).reshape(-1, 1)
        not_infinity = np.where((by_reshape != np.inf) & (by_reshape != -np.inf))[0]

        infitos = np.where(by_reshape == np.inf)[0]
        infitos_neg = np.where(by_reshape == -np.inf)[0]

        if len(not_infinity) != 0:
            by_reshape[infitos] = np.max(by_reshape[not_infinity]) + 1
            by_reshape[infitos_neg] = np.min(by_reshape[not_infinity]) - 1

        else:
            if len(infitos) != 0:
                by_reshape[infitos] = 1

            if len(infitos_neg) != 0:
                by_reshape[infitos_neg] = -1
            # by_reshape[infitos] = 1

        by_minmax = MinMaxScaler().fit_transform(by_reshape)
        return by_minmax

    ####################################
    # Getters functions ################
    ####################################

    def get_measure(self, measure):
        """
        Return the interest measure of each rule
        Parameters
        ----------
        measure

        Returns
        -------

        """

        # Check if measure was defined
        if isinstance(self[0].interest_measures[measure], dict):
            self.update_interest_measures([measure])

        return np.array([rule.get_measure(measure) for rule in self])

    # def update_interest_measures(self, interest_measures=None):
    #     """
    #     Update the interest measures of the rules
    #     Returns
    #     -------
    #
    #     """
    #     A = self.get_sup_antecedent()
    #     B = self.get_sup_consequent()
    #     AB = self.get_support()
    #
    #     self.update_confidence()
    #     self.set_measures(A, B, AB, self.get_n_transactions(), interest_measures)

    def get_measures(self, measures: InterestMeasuresGroup, normalized=None):
        """
        Parameters
        ----------
        measures: InterestMeasuresGroup
            Interest measures to be returned
        normalized: str
            If None, return the interest measures as they are. If "rank", return the rank of each interest measure

        Returns
        -------
        numpy array
            Interest measures of each rule
        """
        # if isinstance(measures, list):
        #     measures = InterestMeasuresGroup('MOs', measures)

        if len(self[0].interest_measures) == 0:
            self.set_measures(self.get_sup_antecedent(),
                              self.get_sup_consequent(),
                              self.get_support(),
                              self.get_n_transactions()[0],
                              measures)

        measures_values = np.array([[rule.get_measure(m) for m in measures.get_measures()] for rule in self])

        if np.isnan(measures_values).any(axis=1).any():
            raise ValueError("There are NaN values in the interest measures")

        if normalized is None:
            return measures_values

        elif normalized == "minmax":
            aux = np.array([self.__scaler_by(m) for m in measures_values.T]).T[0]
            return aux

        elif normalized == "rank":
            rank = ss.rankdata(measures_values, axis=0, method="average")
            return rank / rank.shape[0]

    def get_cover_A(self, data):
        """
        Return index of the rules that cover by the antecedent of the rules
        Parameters
        ----------
        data

        Returns
        -------

        """
        cover_index = list()
        for rule in self:
            cover_index.extend(rule.match_A(data))

        cover_index = np.array(list(set(cover_index)))
        return cover_index

    def get_rank_values(self):
        return np.array([rule.rank_order for rule in self])

    def get_sup_antecedent(self):
        return np.array([rule.sup_antecedent for rule in self])

    def get_sup_consequent(self):
        return np.array([rule.sup_consequent for rule in self])

    def get_support(self):
        return np.array([rule.support for rule in self])

    def get_n_transactions(self):
        return np.array([[rule.n_transactions] for rule in self])

    ####################################
    # Match functions ##################
    ####################################

    def match(self, data):
        """
        Return the index of the rules that match the data instance with antecedent and consequent
        Parameters
        ----------
        data

        Returns
        -------

        """
        return np.array([idx for idx, rule in enumerate(self) if rule.match(data)])

    def antecedent_match(self, instance):
        """
        Return the index of the rules that match the data instance **only** with antecedent
        Parameters
        ----------
        instance

        Returns
        -------

        """
        antecedent_np = np.array([rule.antecedent for rule in self])
        true_antecedent = np.isnan(antecedent_np)

        index = np.where(
            (np.logical_or(true_antecedent, np.equal(instance, antecedent_np))).all(
                axis=1
            )
        )[0]
        return index

    ####################################
    #  Load functions ##################
    ####################################

    @classmethod
    def from_numpy(cls, rules, sup_rule, sup_antecedent, sup_consequent, n_transactions):
        """
        Create a CARs object from numpy arrays
        Parameters
        ----------
        rules
        sup_rule
        sup_antecedent
        sup_consequent
        n_transactions

        Returns
        -------

        """
        data = cls([CAR.from_numpy(rule, sup, a, b, n_transactions) for rule, sup, conf, a, b in
                    zip(rules, sup_rule, sup_antecedent, sup_consequent)])
        return data

    ####################################
    # Export functions #################
    ####################################

    def to_pandas(self, complete=False):
        """
        Return the rules in pandas DataFrame format
        Parameters
        ----------
        complete: bool
            If True, return the interest measures of the rules

        Returns
        -------
        pandas DataFrame
            Rules in DataFrame format

        """

        dataframe = pd.DataFrame(
            [
                {
                    "ID": rule.rid,
                    "Antecedent": rule.antecedent,
                    "Consequent": rule.consequent,
                    "A": rule.sup_antecedent,
                    "B": rule.sup_consequent,
                    "AB": rule.AB,
                    "Support": rule.support,
                    "Confidence": rule.confidence,
                }
                for rule in self
            ]
        )

        if complete:
            interest_measures = pd.DataFrame(
                [rule.interest_measures for rule in self],
                columns=self[0].interest_measures.keys(),
            )
            return pd.concat([dataframe, interest_measures], axis=1)

        return dataframe

    def to_numpy(self):
        """
        Return the rules in numpy array format
        Returns
        -------

        """
        return np.array([rule.rule for rule in self])

    def to_csv(self, path):
        """
        Save the rules in a csv file
        Parameters
        ----------
        path

        Returns
        -------

        """
        self.to_pandas().to_csv(path, index=False)

    def to_excel(self, path):
        """
        Save the rules in a excel file
        Parameters
        ----------
        path

        Returns
        -------

        """
        self.to_pandas().to_excel(path, index=False)

    def save(self, path):
        """
        Save the rules in a pickle file
        Parameters
        ----------
        path

        Returns
        -------

        """
        with open(path, "wb") as file:
            pickle.dump(self, file)


    @classmethod
    def load(cls, path):
        """
        Load the rules from a pickle file
        Parameters
        ----------
        path

        Returns
        -------

        """
        with open(path, "rb") as file:
            return pickle.load(file)