import numpy as np
import pandas as pd
from .LogisticRegression import LogisticRegressionCustom
import itertools


class Search:
    def __init__(self, model: LogisticRegressionCustom):
        self.model = model
        self.best_model = None
        self.best_score = -np.inf
        self.best_params = None

    def _reset(self):
        self.best_model = None
        self.best_score = -np.inf
        self.best_params = None

    def grid_search(self, X_train, y_train, X_test, y_test, param_grid: dict):
        self._reset()

        # key will be the parameter name, value will be the list of values to try
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            # re-initialize model with new params
            model = type(self.model)(**params)
            model.run_logistic_regression(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_params = params

        return self.best_model, self.best_score, self.best_params

    def random_search(self, X_train, y_train, X_test, y_test, param_distributions: dict, n_iter=10, random_state=None):
        self._reset()

        if random_state is not None:
            np.random.seed(random_state)

        for _ in range(n_iter):
            params = {k: np.random.choice(v) for k, v in param_distributions.items()}
            # re-initialize model with new params
            model = type(self.model)(**params)
            model.run_logistic_regression(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_params = params

        return self.best_model, self.best_score, self.best_params