import sys

import numpy as np
import scipy
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale
from sklearn.utils.validation import check_is_fitted

import pywt
from ml_project.models.utils import check_X_tuple, fixBaseline


class Hstack(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, keep='both'):
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Concatenating all features...")
        sys.stdout.flush()
        X1, X2 = check_X_tuple(X)
        if self.keep == 'both':
            if len(X2) == 0:
                newX = X1
            else:
                newX = np.hstack((X1, X2))
        elif self.keep == 'left':
            newX = X1
        elif self.keep == 'right':
            newX = X2
        print("New shape after concatenation:", newX.shape)
        sys.stdout.flush()
        newX = np.nan_to_num(newX, copy=False)
        return newX


class CutTimeSeries(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, t):
        self.t = t

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["t"])
        X1, X2 = check_X_tuple(X)
        print("Shape before cutting: ", X1.shape)
        X1 = X1[:, 100:self.t + 100]
        print("Shape after cutting: ", X1.shape)
        sys.stdout.flush()
        return (X1, X2)


class DiscreteWavelets(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        print("Shape before DWT: ", X1.shape)
        cA, _ = pywt.dwt(X1[0], 'haar')
        new_length = len(cA)
        for i in range(X1.shape[0]):
            cA, _ = pywt.dwt(X1[i], 'haar')
            X1[i, 0:new_length] = cA
        X1 = X1[:, 0:new_length]
        print("Shape after DWT: ", X1.shape)
        sys.stdout.flush()
        return (X1, X2)


class ContinuousWavelets(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, start=1, end=10):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        print("Shape before DWT: ", X1.shape)
        cA, _ = pywt.cwt(X1[0], np.arange(self.start, self.end), 'gaus1')
        cA = np.ravel(cA)
        new_length = len(cA)
        X1new = np.empty((X1.shape[0], new_length))
        for i in range(X1.shape[0]):
            cA, _ = pywt.cwt(X1[i], np.arange(self.start, self.end), 'gaus1')
            cA = np.ravel(cA)
            X1new[i, :] = cA
        print("Shape after DWT: ", X1new.shape)
        sys.stdout.flush()
        return (X1new, X2)


class ScaleSamples(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        X1 = scale(
            X1,
            axis=1,
            with_mean=self.with_mean,
            with_std=self.with_std,
            copy=False)
        return (X1, X2)


class MedianFilter(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        for i in range(X1.shape[0]):
            X1[i] = scipy.signal.medfilt(X1[i], kernel_size=self.kernel_size)
        return (X1, X2)


class BaseLineWanderFix(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Fixing baseline wander...")
        sys.stdout.flush()
        X1, X2 = check_X_tuple(X)
        for i in range(X1.shape[0]):
            X1[i] = fixBaseline(X1[i], sampling_rate=self.sampling_rate)
            if i % 300 == 0:
                print("Processing sample:", i)
                sys.stdout.flush()
        print("Baseline fixed...")
        sys.stdout.flush()
        return (X1, X2)


class DownsampleSignal(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, down_factor=3, initial_upsampling=5):
        self.down_factor = down_factor
        self.initial_upsampling = initial_upsampling

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        up = self.initial_upsampling
        down = up * self.down_factor
        new_length = len(scipy.signal.resample_poly(X1[0], up, down))
        for i in range(X1.shape[0]):
            X1[i, 0:new_length] = scipy.signal.resample_poly(X1[i], up, down)
        return (X1[:, 0:new_length], X2)
