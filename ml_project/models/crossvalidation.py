import importlib
import numpy as np
import sklearn as skl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import classification_report


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


class GridSearchCV(skl.model_selection.GridSearchCV):
    """docstring for GridSearchCV"""
    def __init__(self, est_class, est_params, param_grid, cv=None, n_jobs=1):
        self.estimator = est_class(est_params)
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.est_params = est_params
        self.est_class = est_class
        super(GridSearchCV, self).__init__(self.estimator, param_grid, cv=cv,
            n_jobs=n_jobs)

    def fit(self, X, y):
        super(GridSearchCV, self).fit(X, y)
        print(self.cv_results_)

class Pipeline(skl.pipeline.Pipeline):
    """docstring for Pipeline"""
    def __init__(self, class_list):
        self.class_list = class_list
        self.steps = self.load_steps(class_list)
        super(Pipeline, self).__init__(self.steps)
        
    def load_steps(self, class_list):
        steps = []
        for dict_ in class_list:
            name = dict_["class"].__name__
            if "params" in dict_:
                params = dict_["params"]
                steps.append( ( name, dict_["class"](**params) ) )
            else:
                steps.append( ( name, dict_["class"]() ) )
        return steps