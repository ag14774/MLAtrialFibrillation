import sys

import matplotlib.pyplot as plt
import scipy
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import (bandpass_filter, calc_refractory_period,
                                     detect_qrs, effective_process_first_only,
                                     find_best_windows)


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


class CutWindowWithMaxQRS(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 sampling_rate=300,
                 window_size=1000,
                 num_of_windows=300,
                 random_state=None,
                 process_first_only=None):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.process_first_only = process_first_only
        self.random_state = random_state
        self.num_of_windows = num_of_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # After process_first_only elements, other features begin
        # Treat only first process_first_only features as signal
        process_first_only = effective_process_first_only(
            X[0], self.process_first_only)

        if (process_first_only < self.window_size):
            raise Exception(("Window size cannot be larger ",
                             "than the length of the signal"))

        refractory_period = calc_refractory_period(self.sampling_rate)
        for j in range(X.shape[0]):
            if self.random_state is None:
                random_state = check_random_state(self.random_state)
            else:
                random_state = check_random_state(self.random_state + j)
            indices_to_check = sample_without_replacement(
                process_first_only - self.window_size,
                self.num_of_windows,
                random_state=random_state)
            print(indices_to_check.shape)
            max_i = 0
            max_qrs = 0
            for i in indices_to_check:
                peaks, _ = detect_qrs(X[j, i:i + self.window_size],
                                      self.sampling_rate)
                num_of_peaks = len(peaks)
                if num_of_peaks > max_qrs:
                    max_qrs = num_of_peaks
                    max_i = i
            X[j, 0:self.window_size] = X[j, max_i:max_i + self.window_size]
            print(max_qrs, max_i)
            # plt.plot(X[j, 0:self.window_size])
            # plt.show()
        return X[:, 0:self.window_size]


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
