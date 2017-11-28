import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# IDEA: CHOOSE A WINDOW THAT MAXIMIZES THIS METRIC
# def autocorr(x):
#     n = x.size
#     norm = (x - np.mean(x))
#     result = np.correlate(norm, norm, mode='same')
#     acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
#     lag = np.abs(acorr).argmax() + 1
#     r = acorr[lag-1]
#     if np.abs(r) > 0.5:
#       print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
#     else:
#       print('Appears to be not autocorrelated')
#     return r, lag


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
