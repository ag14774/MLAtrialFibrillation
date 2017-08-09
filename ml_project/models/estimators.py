import sklearn as skl
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""
    def __init__(self, arg):
        super(KernelEstimator, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)
        return np.dot(X, self.coef_) + self.y_mean