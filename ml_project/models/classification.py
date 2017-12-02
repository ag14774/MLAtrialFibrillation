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


class DTWKNeighborsClassifier(BaseEstimator, TransformerMixin):
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
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.radius = radius
        self.QRSList = QRSList
        self.sampling_rate = sampling_rate

        self.kneighborsclassifier = None

    def fit(self, X, y):
        self.QRSList = np.array(self.QRSList)
        base_frequency = 250
        proportionality = self.sampling_rate / base_frequency
        refractory_period = round(120 * proportionality)
        length_of_qrs = 2 * refractory_period
        metric_params = {'radius': self.radius, 'length_of_qrs': length_of_qrs}
        self.kneighborsclassifier = KNeighborsClassifier(
            self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=fastdtwQRS,
            metric_params=metric_params,
            n_jobs=self.n_jobs)
        print("Training data QRS detection...shape:", X.shape)
        sys.stdout.flush()
        X, y = transform_data(X, y, self.QRSList, self.sampling_rate)
        print("Transformed training data:", X.shape)
        sys.stdout.flush()
        return self.kneighborsclassifier.fit(X, y)

    def predict(self, X):
        check_is_fitted(self, ["kneighborsclassifier"])
        self.QRSList = np.array(self.QRSList)
        print("Testing data QRS detection...shape", X.shape)
        sys.stdout.flush()

        X, _ = transform_data(X, None, self.QRSList, self.sampling_rate)

        print("Transformed testing data:", X.shape)
        print("Starting prediction...")
        sys.stdout.flush()
        results = self.kneighborsclassifier.predict(X)
        print(results)

    def score(self, X, y):
        return scorer(self, X, y)
