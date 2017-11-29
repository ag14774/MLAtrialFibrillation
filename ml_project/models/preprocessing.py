import sys

import numpy as np
import scipy
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale
from sklearn.utils.validation import check_array, check_is_fitted


class CutTimeSeries(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, t):
        self.t = t

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["t"])
        print("Shape before cutting: ", X.shape)
        X = check_array(X)
        X = X[:, 0:self.t]
        print("Shape after cutting: ", X.shape)
        sys.stdout.flush()
        return X


class ScaleSamples(BaseEstimator, TransformerMixin):
    """docstring"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = scale(X, axis=1, copy=False)
        return X


class MedianFilter(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i in range(X.shape[0]):
            X[i] = scipy.signal.medfilt(X[i], kernel_size=self.kernel_size)
        return X


class DownsampleSignal(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, down_factor=3, initial_upsampling=5):
        self.down_factor = down_factor
        self.initial_upsampling = initial_upsampling

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        up = self.initial_upsampling
        down = up * self.down_factor
        new_length = len(scipy.signal.resample_poly(X[0], up, down))
        for i in range(X.shape[0]):
            X[i, 0:new_length] = scipy.signal.resample_poly(X[i], up, down)
        return X[:, 0:new_length]
