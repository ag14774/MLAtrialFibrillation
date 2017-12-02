from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from ml_project.models.utils import transform_data


class ExtractQRS(BaseEstimator, TransformerMixin):
    """Random Selection of features"""

    def __init__(self, sampling_rate, random_state=None):
        self.sampling_rate = sampling_rate
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X, _ = transform_data(
            X,
            None,
            QRSList=None,
            random_state=self.random_state,
            sampling_rate=self.sampling_rate)
        return X
