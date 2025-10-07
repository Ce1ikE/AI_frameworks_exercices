import numpy as np
import pandas as pd

class StdScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: pd.DataFrame):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)

    def transform(self, X: pd.DataFrame):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StdScaler instance is not fitted yet. Call 'fit' with appropriate data before using this method.")
        # this is basically (X - mu) / std_dev --> z-score
        # in 
        #   f(x) =  1/sqrt(2pi*sigma^2) * e^(-(x-mu)^2 / 2sigma^2)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)