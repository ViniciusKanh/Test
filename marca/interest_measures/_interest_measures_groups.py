import numpy as np
from sklearn.preprocessing import MinMaxScaler

class InterestMeasuresGroup:
    def __init__(self, name: str, measures, weights=None):
        """

        Parameters
        ----------
        name: str
            Name of the group
        measures: list or str of measure
            List of measures
        weights: list
            List of weights

        Return
        ------
        interest_measures_group: InterestMeasuresGroup
            InterestMeasuresGroup object with name, measures and weights

        Exemple
        -------

        >>> measures = ['supp', 'conf', 'lift']
        >>> weights = [0.5, 0.3, 0.2]
        >>> interest_measures_group = InterestMeasuresGroup(name='group1', measures=measures, weights=weights)
        >>> interest_measures_group.get_measures()
        ['supp', 'conf', 'lift']

        >>> interest_measures_group.get_weights()
        array([0.5, 0.3, 0.2])

        >>> interest_measures_group.set_weights([0.1, 0.2, 0.3])
        >>> interest_measures_group.get_weights()
        array([0.1, 0.2, 0.3])

        """

        self.name = name
        if isinstance(measures, str):
            measures = [measures]

        if isinstance(measures, tuple):
            measures = list(measures)

        if isinstance(measures, list):
            if len(measures) == 0:
                raise ValueError("Measures can't be empty")

        self.measures = measures
        if weights is None:
            self.weights = np.array(
                [1 for _ in range(len(self.measures))]
            )
        else:
            self.weights = weights

    def set_weights(self, weights):
        self.weights = weights

    def set_measures(self, measures):
        self.measures = measures

    def __iter__(self):
        return iter(self.measures)

    def __call__(self, *args, **kwargs):
        return self

    def get_measures(self):
        return self.measures

    def get_weights(self):
        return self.weights

    def get_top_k_measures(self, k):
        """
        Return top k measures based on weights
        Parameters
        ----------
        k: int
            Number of measures to return

        Returns
        -------
        top_k_measures: list
        """
        return np.argsort(self.weights)[::-1][:k]

    def normalize_weights(self, negative=False):
        # Invert weights to keep order + is better than -
        if negative:
            self.weights = -1*self.weights

        # If have infinite in weights, replace by 0
        if np.isinf(self.weights).any():
            #print(dict(zip(self.measures, self.weights)))
            self.weights[np.isinf(self.weights)] = 0

        self.non_normalized_weights = self.weights
        # Minmax sklearn normalization
        scaler = MinMaxScaler()
        self.weights = np.around(scaler.fit_transform(self.weights.reshape(-1, 1)).reshape(-1), 7)


    def __repr__(self):
        return self.name


