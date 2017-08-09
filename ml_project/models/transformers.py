import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.random import sample_without_replacement

class RandomSelection(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        self.components = None

    def fit(self, X, y=None):   
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):       
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new