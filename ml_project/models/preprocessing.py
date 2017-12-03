import sys

import scipy
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import bandpass_filter, find_best_windows


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
        X = X[:, 100:self.t + 100]
        print("Shape after cutting: ", X.shape)
        sys.stdout.flush()
        return X


class CutBestWindow(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, size=1000):
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Shape before cutting: ", X.shape)
        X = find_best_windows(X, self.size)
        print("Shape after cutting: ", X.shape)
        sys.stdout.flush()
        return X


class ScaleSamples(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = scale(
            X,
            axis=1,
            with_mean=self.with_mean,
            with_std=self.with_std,
            copy=False)
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


class BandpassFilter(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 skip=False,
                 lowcut=0.0,
                 highcut=15.0,
                 sampling_rate=300,
                 filter_order=1):
        self.skip = skip
        self.lowcut = lowcut
        self.highcut = highcut
        self.sampling_rate = sampling_rate
        self.filter_order = filter_order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.skip is True:
            return X
        for i in range(X.shape[0]):
            X[i] = bandpass_filter(
                X[i],
                lowcut=self.lowcut,
                highcut=self.highcut,
                signal_freq=self.sampling_rate,
                filter_order=self.filter_order)
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
