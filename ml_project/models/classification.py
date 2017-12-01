import sys
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import DTWDistance, LB_Keogh, knn, scorer


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
                 accuracy='window',
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
        # accuracy = 'exact' -> 'radius' is ignored
        # accuracy = 'approximate' -> 'radius' is the level of accuracy
        # accuracy = 'window' -> 'radius' is the window
        self.accuracy = accuracy
        self.radius = radius
        # TODO: move this to predict
        self.metric_params = {'accuracy': accuracy, 'radius': radius}
        self.metric = DTWDistance

    def fit(self, X, y):
        print("Fitting on training data with shape: ", X.shape)
        sys.stdout.flush()
        if self.n_neighbors == 1 and self.accuracy == 'window':
            self.train = X
            self.train_labels = y
            return self
        else:
            return super(DTWKNeighborsClassifier, self).fit(X, y)

    def predict(self, X):
        print("Predicting data with shape: ", X.shape)
        sys.stdout.flush()
        if self.n_neighbors == 1 and self.accuracy == 'window':
            self.accuracy = 1
            return knn(
                self.train,
                self.train_labels,
                X,
                self.radius,
                accuracy=self.accuracy)
        else:
            self.metric_params = {
                'accuracy': self.accuracy,
                'radius': self.radius
            }
            return super(DTWKNeighborsClassifier, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)
