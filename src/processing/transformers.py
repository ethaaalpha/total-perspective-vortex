from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
import scipy.linalg as lg
import numpy as np

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    # def fit(self, X, y):
    #     n_classes = np.unique(y)

    #     self.csp = CSP(self.n_components, log=True)
    #     self.csp.fit(X, y)
    #     return self

    # def transform(self, X):
    #     return self.csp.transform(X)

class CSPCustom(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes = np.unique(y)

        covs = [self.__get_class_data(X, y, i) for i in np.unique(y)]

        D = np.concatenate(covs)
        E, U = np.linalg.eigh(D)
        W = np.dot(np.diag(np.sqrt(1/(E + 1e-6))), U.T)
        
        # Whiten data
        X_white = np.dot(X, W.T)
        
        # Calculate spatial filters
        S1 = np.dot(np.dot(covs[0], W.T), W)
        S2 = np.dot(np.dot(covs[1], W.T), W)
        E, U = np.linalg.eigh(S1, S1 + S2)
        W_csp = np.dot(U.T, W)
        
        # Apply spatial filters
        X_csp = np.dot(X_white, W_csp.T)
        
        # Select top CSP components
        self.filters_ = X_csp[:, :self.n_components]

        return self

    def transform(self, X):
        n_trials = X.shape[0]
        
        X_transformed = np.zeros((n_trials, self.n_components))
        
        for i in range(n_trials):
            trial = X[i, :, :]  # Shape (n_channels, n_timepoints)
            trial_flattened = trial.flatten()
            X_transformed[i] = trial_flattened @ self.filters_
        
        return X_transformed
    
    def __get_class_data(self, X, y, class_label):
        """Extract a specific class from X"""
        return X[y == class_label]