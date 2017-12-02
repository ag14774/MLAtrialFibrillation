import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import fastdtwQRS, scorer, transform_data


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
                 leaf_size=30,
                 p=2,
                 n_jobs=1,
                 radius=1,
                 QRSList=[3, 7],
                 sampling_rate=300,
                 **kwargs):
        super(DTWKNeighborsClassifier, self).__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
            **kwargs)
        self.radius = radius
        self.QRSList = QRSList
        self.sampling_rate = sampling_rate

    def fit(self, X, y):
        print("Training data QRS detection...shape:", X.shape)
        sys.stdout.flush()
        X, y = transform_data(X, y, self.QRSList, self.sampling_rate)
        print("Transformed training data:", X.shape)
        sys.stdout.flush()
        return super(DTWKNeighborsClassifier, self).fit(X, y)

    def predict(self, X):
        print("Testing data QRS detection...shape", X.shape)
        sys.stdout.flush()

        base_frequency = 250
        proportionality = self.sampling_rate / base_frequency
        refractory_period = round(120 * proportionality)
        length_of_qrs = 2 * refractory_period
        self.metric = fastdtwQRS
        self.metric_params = {
            'radius': self.radius,
            'length_of_qrs': length_of_qrs
        }
        X, _ = transform_data(X, None, self.QRSList, self.sampling_rate)

        print("Transformed testing data:", X.shape)
        print("Starting prediction...")
        sys.stdout.flush()
        return super(DTWKNeighborsClassifier, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)
