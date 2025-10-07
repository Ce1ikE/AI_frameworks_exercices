import numpy as np
import pandas as pd
import pprint

class LinearRegressionCustom:
    def __init__(
        self,
        save_history_every_n_iter: int = 1000,
        num_iterations: int = 100000,
        learning_rate: float = 0.01,
    ):
        # hyperparameters
        self.num_iterations = num_iterations
        # alpha in some books 
        self.learning_rate = learning_rate

        self.theta = None

        self.save_history_every_n_iter = save_history_every_n_iter
        self.history = []

    # ------------------- Linear Regression ------------------- #

    def feed_forward(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # function f(x) = theta_0 * x0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn
        return X.dot(theta)

    def compute_cost(self,y_pred: np.ndarray, y_true: np.ndarray, m_samples: int) -> float:
        # MSE cost function:
        #  J = 1/2m * sum(f(x_i) - y_i)^2
        # where:
        #       f(x_i) = theta_0 * x0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn
        # J == cost or error
        J = 1/(2*m_samples) * np.sum(np.square(y_pred - y_true))
        return J

    def gradient_descent(self,X: np.ndarray,y_pred: np.ndarray, y_true: np.ndarray, theta: np.ndarray, learning_rate: float, m_samples: int) -> np.ndarray:
        # theta_j = theta_j - learning_rate * d/dtheta_j(J(theta))
        # where:
        #       J(theta) = 1/2m * sum(h(x_i) - y_i)^2
        #       d/dtheta_j(J(theta)) = 1/m * sum((h(x_i) - y_i) * x_i_j)
        gradient = (1/m_samples) * X.T.dot(y_pred - y_true)
        theta = theta - learning_rate * gradient
        return theta
    
    def run_linear_regression(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
    ):
        # X comes in as a DataFrame of [m_samples, n_features]
        m_samples , n_features = X.shape
        X = X.to_numpy()
        # y comes in as a Series of [m_samples] (array-like) reshape to [m_samples, 1] (column vector)
        y = y.to_numpy().reshape(-1, 1)
        
        # https://numpy.org/doc/stable/reference/generated/numpy.c_.html
        # so it can be multiplied by theta
        # c_ is for column-wise concatenation
        self.X_b = np.c_[
            # array of [1, 1, 1, ..., 1] with shape (m, 1)
            np.ones((m_samples, 1)),
            # expression to concatenate
            X
        ]
        # adds the prediction for y when x = 0 (bias term) 
        # (the x0 in f(x) = theta_0 * x0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn)
        # we must have as many theta as there are features + 1 (for bias)
        theta = np.zeros([n_features + 1, 1])
        # steps:
        # 1) feed forward to get predictions
        # 2) compute cost function
        # 3) update theta using gradient descent
        for i in range(self.num_iterations):
            predictions = self.feed_forward(self.X_b, theta)
            cost = self.compute_cost(predictions, y, m_samples)
            theta = self.gradient_descent(self.X_b, predictions, y, theta, self.learning_rate, m_samples)

            if i % self.save_history_every_n_iter == 0 or i == self.num_iterations - 1:
                self.history.append({
                    "iteration": i, 
                    "cost": cost, 
                    "thetas": [theta[j][0] for j in range(theta.shape[0])]
                })
                pprint.pprint(self.history[-1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.to_numpy()
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.feed_forward(X_b, self.history[-1]['thetas'])
