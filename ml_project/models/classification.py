import numpy as np
from ml_project.models.utils import probs2labels
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class SupportVectorClassifier(SVC):
    """docstring for SVM"""
    def fit(self, X, y):
        y = probs2labels(y)
        super(SupportVectorClassifier, self).fit(X, y)
    
    def predict_proba(self, X):
        T = super(SupportVectorClassifier, self).predict_proba(X)
        n_samples, _ = T.shape
        return np.hstack((T, np.zeros((n_samples, 1))))


class LinearDiscriminant(LinearDiscriminantAnalysis):
    """docstring for LinearDiscriminant"""
    def fit(self, X, y):
        y = probs2labels(y)
        super(LinearDiscriminant, self).fit(X, y)
    
    def predict_proba(self, X):
        T = super(LinearDiscriminant, self).predict_proba(X)
        n_samples, _ = T.shape
        return np.hstack((T, np.zeros((n_samples, 1))))