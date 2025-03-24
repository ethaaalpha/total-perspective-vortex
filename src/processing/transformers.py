from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.csp = CSP(self.n_components, log=True)
        self.csp.fit(X, y)
        return self

    def transform(self, X):
        return self.csp.transform(X)
