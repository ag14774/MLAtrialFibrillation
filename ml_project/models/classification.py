import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import scorer


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


class SVMClassifier(SVC):
    """docstring"""

    def __init__(self,
                 C=1.0,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=False,
                 tol=0.001,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None):
        super(SVMClassifier, self).__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)

    def fit(self, X, y=None):
        print("Fitting SVC on data with shape", X.shape)
        sys.stdout.flush()
        return super(SVMClassifier, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(SVMClassifier, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)


class RandomForest(RandomForestClassifier):
    """docstring"""

    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForest, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

    def fit(self, X, y=None):
        print("Fitting random forest on data with shape", X.shape)
        sys.stdout.flush()
        return super(RandomForest, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(RandomForest, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)
