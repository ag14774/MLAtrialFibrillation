import sys

import numpy as np
from scipy.stats import signaltonoise
from sklearn.base import BaseEstimator, TransformerMixin

from ml_project.models.utils import (autocorr, biosspyX, check_X_tuple,
                                     extract_data, template_min_length,
                                     featurevector)


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        processed = biosspyX(
            X1, sampling_rate=self.sampling_rate, show=False, verbose=True)

        template_length = template_min_length(processed)
        _, temp = featurevector(
            processed[0],
            sampling_rate=self.sampling_rate,
            min_length=template_length)
        new_length = len(temp)
        print(template_length, new_length)
        all_features = np.empty((X1.shape[0], new_length))
        for i in range(X1.shape[0]):
            filtered_signal, features = featurevector(
                processed[i],
                sampling_rate=self.sampling_rate,
                min_length=template_length)

            assert all_features.shape[1] == len(features)
            all_features[i, :] = features
            X1[i, :] = filtered_signal

            if i % 300 == 0:
                print("Extracting features from sample", i)
                sys.stdout.flush()

        X2 = X2.reshape(X1.shape[0], -1)
        X2 = np.hstack((X2, all_features))
        return (X1, X2)


class SignalToNoiseRatio(BaseEstimator, TransformerMixin):
    """docstring"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Calculating signal to noise ratio...")
        sys.stdout.flush()
        X1, X2 = check_X_tuple(X)
        X2 = X2.reshape(X1.shape[0], -1)
        snr = signaltonoise(X1, axis=1)
        snr = snr.reshape(X1.shape[0], -1)
        X2 = np.hstack((X2, snr))
        X2 = X2.reshape(X1.shape[0], -1)
        return (X1, X2)


class Autocorrelogram(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, first_fraction=1.0):
        self.first_fraction = first_fraction

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Computing autocorrelogram...")
        sys.stdout.flush()
        X1, X2 = check_X_tuple(X)
        upto = round(X1.shape[1] * self.first_fraction)
        _, _, corr = autocorr(X1[0, :upto])
        newfeatures = np.empty((X1.shape[0], len(corr)), dtype=float)
        for i in range(X1.shape[0]):
            _, _, corr = autocorr(X1[i, :upto])
            newfeatures[i, :] = corr
        print(newfeatures.shape)
        X2 = X2.reshape(X1.shape[0], -1)
        X2 = np.hstack((X2, newfeatures))
        X2 = X2.reshape(X1.shape[0], -1)
        return (X1, X2)
