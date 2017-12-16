import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from ml_project.models.utils import scorer
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier, plot_importance


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
                 min_impurity_split=1e-07,
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


class AdaBoost(AdaBoostClassifier):
    """docstring"""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME.R',
                 random_state=None):
        super(AdaBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state)

    def fit(self, X, y=None):
        print("Fitting ada boost on data with shape", X.shape)
        sys.stdout.flush()
        return super(AdaBoost, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(AdaBoost, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)


class GradientBoosting(GradientBoostingClassifier):
    """docstring"""

    def __init__(self,
                 loss='deviance',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_split=1e-7,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto'):
        super(GradientBoosting, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start,
            presort=presort)

    def fit(self, X, y=None):
        print("Fitting gradient boosting on data with shape", X.shape)
        sys.stdout.flush()
        return super(GradientBoosting, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(GradientBoosting, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)


class MLP(MLPClassifier):
    """docstring"""

    def __init__(self,
                 hidden_layer_sizes=(100, ),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate='constant',
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=0.0001,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08):
        super(MLP, self).__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon)

    def fit(self, X, y=None):
        print("Fitting MLP on data with shape", X.shape)
        sys.stdout.flush()
        return super(MLP, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(MLP, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)


class XGB(XGBClassifier):
    """docstring"""

    def __init__(self,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 silent=True,
                 objective='binary:logistic',
                 nthread=1,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 base_score=0.5,
                 seed=0,
                 missing=None,
                 **kwargs):
        super(XGB, self).__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            silent=silent,
            objective=objective,
            nthread=nthread,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            seed=seed,
            missing=missing,
            **kwargs)

    def fit(self, X, y=None):
        print("Fitting XGB on data with shape", X.shape)
        sys.stdout.flush()
        super(XGB, self).fit(X, y)
        plot_importance(self)
        plt.show()
        return self

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(XGB, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)


class KNN(KNeighborsClassifier):
    """docstring"""

    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=1,
                 **kwargs):
        super(KNN, self).__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs)

    def fit(self, X, y=None):
        print("Fitting KNN on data with shape", X.shape)
        sys.stdout.flush()
        return super(KNN, self).fit(X, y)

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return super(KNN, self).predict(X)

    def score(self, X, y):
        return scorer(self, X, y)


class RFEXGB(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 n_features_to_select=None,
                 step=1,
                 verbose=0,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 silent=True,
                 objective='binary:logistic',
                 nthread=1,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 base_score=0.5,
                 seed=0,
                 missing=None):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.seed = seed
        self.missing = missing
        self.rfe = None

    def fit(self, X, y=None):
        print("Fitting RFE-XGB on data with shape", X.shape)
        sys.stdout.flush()
        base_estimator = XGB(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            silent=self.silent,
            objective=self.objective,
            nthread=self.nthread,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            seed=self.seed,
            missing=self.missing)
        self.rfe = RFE(
            base_estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step,
            verbose=self.verbose)
        self.rfe.fit(X, y)
        return self

    def predict(self, X, y=None):
        print("Predicting data with shape", X.shape)
        sys.stdout.flush()
        return self.rfe.predict(X)

    def score(self, X, y):
        return scorer(self.rfe, X, y)
