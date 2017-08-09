from sklearn.model_selection import GridSearchCV
import pandas as pd

class GridSearchCV(GridSearchCV):
    """docstring for GridSearchCV"""
    def __init__(self, est_class, est_params, param_grid, cv=None, n_jobs=1):
        self.est_class = est_class
        self.est_params = est_params
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.estimator = est_class(est_params)
        super(GridSearchCV, self).__init__(self.estimator, param_grid, cv=cv,
            n_jobs=n_jobs)

    def save(self, save_path):
        data = {
            "best_params_" : self.best_params_,
            "mean_test_score" : self.cv_results_["mean_test_score"],
            "std_test_score" : self.cv_results_["std_test_score"],
        }
        df = pd.DataFrame.from_dict(pd.io.json.json_normalize(data))
        df.to_csv(save_path+"GridSearchCV.csv")