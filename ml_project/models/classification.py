import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ml_project.models.utils import probs2labels
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


class SupportVectorClassifier(SVC):
    """docstring for SVM"""
    def fit(self, X, y):
        y = probs2labels(y)
        super(SupportVectorClassifier, self).fit(X, y)
    
    def predict_proba(self, X):
        T = super(SupportVectorClassifier, self).predict_proba(X)
        n_samples, _ = T.shape
        return T #np.hstack((T, np.zeros((n_samples, 1))))


class LinearDiscriminant(LinearDiscriminantAnalysis):
    """docstring for LinearDiscriminant"""
    def fit(self, X, y):
        y = probs2labels(y)
        super(LinearDiscriminant, self).fit(X, y)
    
    def predict_proba(self, X):
        T = super(LinearDiscriminant, self).predict_proba(X)
        n_samples, _ = T.shape
        return T #np.hstack((T, np.zeros((n_samples, 1))))


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for LinearDiscriminant"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))