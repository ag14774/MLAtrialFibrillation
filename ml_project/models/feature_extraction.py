import sys

import numpy as np
from scipy.stats import signaltonoise
from sklearn.base import BaseEstimator, TransformerMixin

from ml_project.models.utils import autocorr, check_X_tuple


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X1, X2 = check_X_tuple(X)
        return (X1, X2)


class SignalToNoiseRatio(BaseEstimator, TransformerMixin):
    """docstring"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Calculating signal to noise ratio...")
        sys.stdout.flush()
        X1, X2 = check_X_tuple(X)
        snr = signaltonoise(X1, axis=1)
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
