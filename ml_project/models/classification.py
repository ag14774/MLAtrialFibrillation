from ml_project.models.utils import probs2labels
from sklearn.svm import SVC

class SupportVectorClassifier(SVC):
    """docstring for SVM"""
    def fit(self, X, y):
        y = probs2labels(y)
        super(SupportVectorClassifier, self).fit(X, y)
        