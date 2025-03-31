from sklearn.base import BaseEstimator, TransformerMixin
import scipy.linalg as lg
import numpy as np

class CSPCustom(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        cov_matrix = self.__get_covariance_matrix(X, y)
        eigh_values, eigh_vectors = lg.eigh(cov_matrix[1], np.sum(cov_matrix, axis=0))

        # to get in ascending order
        sorted_indexes = np.argsort(np.abs(eigh_values - 0.5))[::-1]
        
        eigh_vectors = eigh_vectors[:, sorted_indexes]

        # transpose as in subject (n_components, n_channels)
        # this will be usefull when applying the filter then !
        W = eigh_vectors.T 

        # extract the filters based on the number of components
        self.filters_ = W[:self.n_components]
        X = np.asarray([self.filters_ @ epoch for epoch in X])

        # better than without
        X = (X**2).mean(axis=2)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray):
        X = np.asarray([self.filters_ @ epoch for epoch in X])
        X = (X**2).mean(axis=2)
        X = (X - self.mean_) / self.std_

        return X

    def __get_covariance_matrix(self, X, y):
        # n_epochs, n_channels, n_times
        classes = np.unique(y)
        cov_class = []

        for class_label in classes:
            data = X[y == class_label]
            _, n_channels, _ = data.shape

            # change to n_channels, n_epochs, n_times
            data = data.transpose([1, 0, 2])
            # change to n_channels, n_epochs * n_times
            data = data.reshape(n_channels, -1)

            cov_class.append(np.cov(data))

        return np.stack(cov_class)