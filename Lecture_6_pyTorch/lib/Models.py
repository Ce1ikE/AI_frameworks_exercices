import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class Model:
    def train(self,X,y):
        raise NotImplementedError
    
    def predict(self,X):
        raise NotImplementedError

class CustomModel(Model):
    def __init__(
        self,
        learning_rate = 0.01,
        epochs = 20,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = []

    def feed_forward(
        self,
        X: np.ndarray, 
        theta: np.ndarray
    ) -> np.ndarray:
        # function f(x) = theta_0 * x0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn
        return X.dot(theta)

    # Mean squared error (! loss == cost noted as J)
    #  J = 1/2m * sum(f(x_i) - y_i)^2
    def compute_loss(
        self,
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        m_samples: int
    ) -> float:
        J = 1/(2*m_samples) * np.sum(np.square(y_pred - y_true))
        return J

    # Here we need to calculate the gradient of the loss-function ourselves.
    # Loss == J = 1/2m * sum(f(x_i) - y_i)^2
    # dLoss/dw == d/dtheta_j(J(theta)) =  1/m * sum(h(x_i) - y_i * x_i_j)
    def gradient_descent(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray,
        y_true: np.ndarray,
        theta: np.ndarray, 
        m_samples: int
    ) -> np.ndarray:
        gradient = (1/m_samples) * X.T.dot(y_pred - y_true)
        theta = theta - self.learning_rate * gradient
        return theta, gradient

    def train(
        self,
        X: np.ndarray, 
        y: np.ndarray,
    ):
        X = X.reshape(-1, 1)
        m_samples , n_features = X.shape
        y = y.reshape(-1, 1)
        self.X_b = np.c_[
            # array of [1, 1, 1, ..., 1] with shape (m, 1)
            np.ones((m_samples, 1)),
            # expression to concatenate
            X
        ]
        
        theta = np.zeros([n_features + 1, 1])
        for epoch in range(self.epochs + 1):
            # predict = forward pass
            y_pred = self.feed_forward(self.X_b, theta)
            # calculate the loss (cost) 
            cost = self.compute_loss(y_pred,y,m_samples)
            # perform GD
            theta, gradient = self.gradient_descent(self.X_b, y_pred, y, theta, m_samples)

            self.history.append({
                "epoch": epoch,
                "cost": cost,
                "thetas": [theta[j][0] for j in range(theta.shape[0])],
                "gradient": np.linalg.norm(gradient)
            })

        return pd.DataFrame(self.history)

    def predict(
        self,
        X: np.ndarray, 
    ):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.array(self.history[-1]['thetas']).reshape(-1, 1)
        return self.feed_forward(X_b, theta)




class PyTorchModel(Model):
    def __init__(
        self,
        learning_rate = 0.01,
        epochs = 20,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = []

    def train(
        self,
        X: torch.Tensor, 
        y: torch.Tensor,
    ):
        # [m, n] and [m, 1]
        if y.dim() == 1:
            y = y.view(-1, 1)

        m_samples , n_features = X.shape

        # ---------- model ---------- #
        self.model = nn.Linear(n_features, 1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs + 1):
            # predict = forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            # calculate the loss (cost) 
            cost = self.loss_fn(y_pred,y)
            # backpropagation 
            cost.backward()
            # update weights
            self.optimizer.step()

            with torch.no_grad():
                thetas = [p.detach().numpy().copy() for p in self.model.parameters()]
                self.history.append({
                    "epoch": epoch,
                    "cost": cost.item(),
                    "thetas": thetas,
                })

        return pd.DataFrame(self.history)
    
    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            return self.model(X)
