import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from ml_project.models.utils import (check_X_tuple, detect_qrs,
                                     squared_diff_minmax)


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        return (X1, X2)
