class Extract:
    def __call__(self, X, y):
        """
        Extract rules from dataset X and target y
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples,)
        :return: CARs object with the extracted rules
        """
        pass

    def _validate_data(self, X, y):
        """
        Validate the input data
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples,)
        """
        pass

    def _transform_data(self, X, y):
        """
        Transform the input data
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples,)
        """

        X = X.astype(str)
        y = y.astype(str)

        return X, y