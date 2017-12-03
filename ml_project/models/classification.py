import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import scorer


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


class SVMClassifier(SVC):
    """docstring"""

    def __init__(self,
                 C=1.0,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=False,
                 tol=0.001,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None):
        super(SVMClassifier, self).__init__(
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='auto',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            random_state=None)

    def fit(self, X, y=None):
        print("Fitting SVC on data with shape", X.shape)
        sys.stdout.flush()
        return super(SVMClassifier, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(SVMClassifier, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)
