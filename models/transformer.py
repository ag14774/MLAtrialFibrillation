import numpy as np
import sklearn as skl
import pickle
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.random import sample_without_replacement
from sklearn.pipeline import Pipeline
from numpy.testing import assert_equal
import subprocess


class RandomSelection(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        self.components = None

    def fit(self, X, y=None):
        
        X = check_array(X)
        n_samples, n_features = X.shape
        """
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0, got %s"
                                         % self.n_components)
        elif self.n_components > n_features:
            warnings.warn("The number of components is higher than the number of"
                          " features: n_features < n_components (%s < %s)."
                          "Setting n_components = n_features."
                          "The dimensionality of the problem will not be reduced."
                          % (n_features, self.n_components),
                          DataDimensionalityWarning)
            self.n_components = n_features
        """
        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        """
        assert_equal(
            self.components.shape,
            (self.n_components,),
            err_msg=("An error has occurred: The self.components vector does "
                     " not have the proper shape."))
        """
        return self

    def transform(self, X, y=None):
        
        X = check_array(X)
        n_samples, n_features = X.shape
        """
        if self.components is None:
            raise NotFittedError("No random selection has been fit.")

        if n_features < self.components.shape[0]:
            raise ValueError("Impossible to perform selection:"
                "X has less features than should be selected."
                "(%s < %s)" % (n_features, self.components.shape[0]))
        """
        X_new = X[:, self.components]
        """
        assert_equal(
            X_new.shape,
            (n_samples, self.n_components),
            err_msg=("An error has occurred: The transformed X does "
                     "not have the proper shape."))
        """
        return X_new

class MyPipeline(Pipeline):
    """docstring for MyPipeline"""
    def __init__(self, arg):
        super(MyPipeline, self).__init__()
        self.arg = arg
        

def submit_csv(csv, message):
    subprocess.run(["kg", "submit", csv, "-m", message])

def submit_archived(timestamp):
    pass


if __name__=="__main__":
    make_submission("data/sampleSubmission_1.csv", "hallo!")