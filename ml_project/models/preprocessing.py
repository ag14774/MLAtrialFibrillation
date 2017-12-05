import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_is_fitted

from ml_project.models.utils import (bandpass_filter, check_X_tuple,
                                     detect_qrs, isolate_qrs)


class Hstack(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Concatenating all features...")
        sys.stdout.flush()
        X1, X2 = check_X_tuple(X)
        if len(X2) == 0:
            newX = X1
        else:
            newX = np.hstack(X1, X2)
        print("New shape after concatenation:", newX.shape)
        sys.stdout.flush()
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


class CutWindowWithMaxQRS(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 sampling_rate=300,
                 window_size=1000,
                 num_of_windows=300,
                 random_state=None):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        self.num_of_windows = num_of_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.window_size % 2 != 0:
            raise Exception("Only even windows are allowed")
        X1, X2 = check_X_tuple(X)
        print("Cutting window with max QRS:", X1.shape)
        sys.stdout.flush()

        if (X1.shape[1] < self.window_size):
            raise Exception(("Window size cannot be larger ",
                             "than the length of the signal"))

        max_qrs_array = np.empty((X1.shape[0]))
        for j in range(X1.shape[0]):
            if self.random_state is None:
                random_state = check_random_state(self.random_state)
            else:
                random_state = check_random_state(self.random_state + j)
            try:
                indices_to_check = sample_without_replacement(
                    X1.shape[1] - self.window_size,
                    self.num_of_windows,
                    random_state=random_state)
            except Exception:
                indices_to_check = sample_without_replacement(
                    X1.shape[1] - self.window_size,
                    X1.shape[1] - self.window_size,
                    random_state=random_state)
            max_i = 0
            max_qrs = 0
            for i in indices_to_check:
                peaks, _ = detect_qrs(X1[j, i:i + self.window_size],
                                      self.sampling_rate)
                num_of_peaks = len(peaks)
                if num_of_peaks > max_qrs:
                    max_qrs = num_of_peaks
                    max_i = i
            X1[j, 0:self.window_size] = X1[j, max_i:max_i + self.window_size]
            max_qrs_array[j] = max_qrs
            if j % 300 == 0:
                print("Processing sample:", j)
                sys.stdout.flush()
            # plt.plot(X[j, 0:self.window_size])
            # plt.show()
        X1 = X1[:, 0:self.window_size]
        print("Cutting completed. New shape:", X1.shape)
        print("Checking and removing noisy spikes in selected windows...")
        sys.stdout.flush()
        for j in range(X1.shape[0]):
            if self.random_state is None:
                random_state = check_random_state(self.random_state)
            else:
                random_state = check_random_state(self.random_state + j)
            new_window_size = self.window_size // 2
            try:
                indices_to_check = sample_without_replacement(
                    X1.shape[1] - new_window_size,
                    self.num_of_windows,
                    random_state=random_state)
            except Exception:
                indices_to_check = sample_without_replacement(
                    X1.shape[1] - new_window_size,
                    X1.shape[1] - new_window_size,
                    random_state=random_state)
            max_i = 0
            max_qrs = 0
            for i in indices_to_check:
                peaks, _ = detect_qrs(X1[j, i:i + new_window_size],
                                      self.sampling_rate)
                num_of_peaks = len(peaks)
                if num_of_peaks > max_qrs:
                    max_qrs = num_of_peaks
                    max_i = i
            # if a smaller window size leads to
            # more QRS complexes then use that
            # two times
            if max_qrs > max_qrs_array[j]:
                new_window = X1[j, max_i:max_i + new_window_size]
                X1[j, 0:new_window_size] = new_window
                X1[j, new_window_size:] = new_window
            if j % 300 == 0:
                print("Re-processing sample:", j)
                sys.stdout.flush()
        return (X1, X2)


class IsolateQRS(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 num_of_qrs=5,
                 keep_full_refractory=False,
                 scale_mode="after",
                 skip_bandpass=True,
                 sampling_rate=300):
        self.num_of_qrs = num_of_qrs
        self.keep_full_refractory = keep_full_refractory
        self.scale_mode = str(scale_mode)
        self.skip_bandpass = skip_bandpass
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        print("Isolating QRS complexes...Shape:", X1.shape)
        sys.stdout.flush()
        # Do an example to get the length
        new_length = len(
            isolate_qrs(
                X1[0],
                num_of_qrs=self.num_of_qrs,
                sampling_rate=self.sampling_rate,
                keep_full_refractory=self.keep_full_refractory,
                scale_mode=self.scale_mode,
                skip_bandpass=self.skip_bandpass))
        X1new = np.empty((X1.shape[0], new_length), dtype=X1.dtype)
        for i in range(X1new.shape[0]):
            X1new[i, :] = isolate_qrs(
                X1[i],
                num_of_qrs=self.num_of_qrs,
                sampling_rate=self.sampling_rate,
                keep_full_refractory=self.keep_full_refractory,
                scale_mode=self.scale_mode,
                skip_bandpass=self.skip_bandpass)
            if i % 300 == 0:
                print("Isolating QRS in sample", i)
                sys.stdout.flush()
        print("All QRS complexes isolated..New shape:", X1new.shape)
        sys.stdout.flush()
        return (X1new, X2)


class CheckQRSNumber(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, minimum_qrs=3, sampling_rate=300):
        self.minimum_qrs = minimum_qrs
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        print("Checking the number of QRS complexes...")
        sys.stdout.flush()
        for i, x in enumerate(X1):
            qrs_peaks, _ = detect_qrs(x, self.sampling_rate)
            if len(qrs_peaks) < self.minimum_qrs:
                print("Low QRS number detected in sample", i)
                print("Total QRS:", len(qrs_peaks))
                sys.stdout.flush()
                plt.plot(x)
                plt.show()
        return X


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
        X1, X2 = check_X_tuple(X)
        if self.skip is True:
            return X
        for i in range(X1.shape[0]):
            X1[i] = bandpass_filter(
                X1[i],
                lowcut=self.lowcut,
                highcut=self.highcut,
                signal_freq=self.sampling_rate,
                filter_order=self.filter_order)
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
