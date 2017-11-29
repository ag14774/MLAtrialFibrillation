import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import DTWDistance, scorer


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


class DTWKNeighborsClassifier(KNeighborsClassifier):
    """docstring"""

    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 accuracy='exact',
                 radius=1,
                 leaf_size=30,
                 p=2,
                 n_jobs=1,
                 **kwargs):
        super(DTWKNeighborsClassifier, self).__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
            **kwargs)
        self.accuracy = accuracy
        self.radius = radius
        self.metric_params = {'accuracy': accuracy, 'radius': radius}
        self.metric = DTWDistance

    def fit(self, X, y):
        print("Fitting on training data with shape: ", X.shape)
        sys.stdout.flush()
        return super(DTWKNeighborsClassifier, self).fit(X, y)

    def predict(self, X):
        self.metric_params = {'accuracy': self.accuracy, 'radius': self.radius}
        print("Predicting data with shape: ", X.shape)
        sys.stdout.flush()
        return super(DTWKNeighborsClassifier, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)
