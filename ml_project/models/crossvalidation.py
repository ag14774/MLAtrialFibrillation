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


class GridSearchCV(skl.GridSearchCV):
    """docstring for GridSearchCV"""
    def __init__(self, arg):
        super(GridSearchCV, self).__init__()
        self.arg = arg
        

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()