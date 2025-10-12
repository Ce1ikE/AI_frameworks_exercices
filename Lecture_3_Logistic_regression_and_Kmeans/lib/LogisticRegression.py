import numpy as np
import pandas as pd
import copy
# logistic regression is technically the same as linear regression but with a different cost function
# and a different prediction function (sigmoid instead of linear)
# I quote from GeeksforGeeks:
# "Logistic regression model transforms the linear regression function continuous value output into categorical value output 
# using a sigmoid function which maps any real-valued set of independent variables input into a value between 0 and 1. 
# This function is known as the logistic function."

class LogisticRegressionCustom:
    def __init__(
        self,
        save_history_every_n_iter: int = 1000,
        num_iterations: int = 100000,
        learning_rate: float = 0.01,
        threshold: float = 0.5,
    ):
        # hyperparameters
        self.num_iterations = num_iterations
        # alpha in some books 
        self.learning_rate = learning_rate
        self.theta = None
        self.classes = None
        self.threshold = threshold

        self.save_history_every_n_iter = save_history_every_n_iter
        self.history = []

    # ------------------- Logistic Regression ------------------- #
    
    # https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/
    # the use of exponential function in sigmoid and softmax is to emphasize larger values and suppress smaller ones
    # this helps in making a clear distinction between classes
    # the sigmoid function is used for binary classification (2 classes)
    # the softmax function is used for multi-class classification (more than 2 classes)
    
    # e.g.: for 3 classes with logits [1.5, 0.5, 0.1]
    # applying exponential function gives us [4.48, 1.65, 1.11] => sum = 7.24
    # then normalizing gives us probabilities 
    # [4.48/7.24, 1.65/7.24, 1.11/7.24] => [0.62, 0.23, 0.15]

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        # sigmoid(z) = 1 / (1 + exp(-z))
        # z is the predicted value of previous iteration
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        # https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/
        # softmax(z_i) = exp(z_i) / sum(exp(z_j) for j = 1 to K)
        # z_i (logit) is the output of the previous iteration for class i
        # K is the number of classes
        # z_j is the logit for each class
        # this ensures that the output is a probability distribution over K classes (i.e. all values are between 0 and 1 and sum to 1)
        # to improve numerical stability, we subtract the max value from z before applying exp
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def feed_forward(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # for logistic regression, the feed forward step is the same as in linear regression:
        #       y_pred = X.dot(theta) + bias
        # we try to approximate (just as in linear regression) the linear combination of inputs and weights (thetas)
        # but the output is not the final prediction, it's just the logits (unbounded real values)
        # we then pass these logits through the sigmoid or softmax function to get probabilities
        linear_pred = X.dot(theta)
        
        # (mxn) . (nxk) = (mxk)
        if len(self.classes) > 2:
            return self.softmax(linear_pred)
        else:
            return self.sigmoid(linear_pred)

    def compute_cost(self, m: int, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # https://www.geeksforgeeks.org/machine-learning/ml-cost-function-in-logistic-regression/
        # for logistic regression, the cost function is not MSE
        # we have two cases:
        
        # 1) for n_categories > 2 (multi-class classification):
        # with cost function called categorical cross-entropy loss:
        # J(theta) = -1/m * sum(y * log(h(x)))
        # where h(x) = softmax(X * theta)
        
        # 2) for n_categories = 2 (binary classification):
        # with cost function also called logistic loss:
        # J(theta) = -1/m * sum(y * log(h(x)) + (1 - y) * log(1 - h(x)))
        # where h(x) = sigmoid(X * theta)
        # where y * log(h(x)) penalizes the model for being wrong when y = 1
        # and (1 - y) * log(1 - h(x)) penalizes the model for being wrong when y = 0
        # or vice versa (doesn't matter much)
        
        # predictions are probabilities between closed interval [0, 1], so we clip them to avoid log(0)
        epsilon = 1e-4

        if len(self.classes) > 2:
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -(1/m) * np.sum(y_true * np.log(y_pred))

        else:
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient_descent(self, m, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, theta: np.ndarray, learning_rate: float) -> np.ndarray:
        # if you run this model once against sklearn's LogisticRegression, 
        # you will see that the values of theta are very similar however not exactly the same everywhere
        # this is due to differences in optimization algorithms, regularization, and other factors
        # such as L2 regularization (Ridge) which is used by default in sklearn's LogisticRegression
        # and the fact that sklearn uses a different solver (like 'lbfgs' or 'liblinear') which can lead to different convergence properties
        # and also the way the bias (intercept) is handled

        # same as in LR, use the predicted probabilities for gradient
        gradient = (1/m) * X.T.dot(y_pred - y_true)
        theta = theta - learning_rate * gradient
        return theta
    
    def run_logistic_regression(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
    ):
        # X comes in as a DataFrame of [m_samples, n_features]
        m_samples , n_features = X.shape
        X = X.to_numpy()
        # y comes in as a Series of [m_samples] (array-like) reshape to [m_samples, 1] (column vector)
        y = y.to_numpy().reshape(-1, 1)
        # extract the different classes from y (k_classes)
        self.classes = np.unique(y)

        # https://numpy.org/doc/stable/reference/generated/numpy.c_.html
        # so it can be multiplied by theta
        # c_ is for column-wise concatenation
        self.X_b = np.c_[
            # array of [1, 1, 1, ..., 1] with shape (m, 1)
            np.ones((m_samples, 1)),
            # expression to concatenate
            X
        ]
        # theta is a matrix of shape (n_features, k_classes) where k_classes is the number of unique classes in y
        # we must have as many theta as there are features + 1 (for bias)
        # and as many columns as there are classes (for multi-class classification)
        # and reshape to 2D array for matrix multiplication
        theta = np.zeros(
            (n_features + 1, len(self.classes))
        )
        # convert y to one-hot encoding if multi-class classification
        y_flat = y.ravel()
        y_encoded = np.eye(
            len(self.classes))[np.searchsorted(self.classes, y_flat)
        ]
        
        if len(self.classes) == 2:
            # for binary classification, we keep y as is (0 and 1)
            y_encoded = y_flat.reshape(-1, 1)
            # for binary classification, we only need one column
            theta = theta[:, [0]].reshape(-1, 1)  

        # steps:
        # 1) feed forward to get predictions
        # 2) compute cost function
        # 3) update theta using gradient descent

        # for every iteration:
        for i in range(self.num_iterations + 1):
            y_pred = self.feed_forward(self.X_b, theta)
            cost = self.compute_cost(m_samples, y_pred, y_encoded)
            theta = self.gradient_descent(m_samples, self.X_b, y_pred, y_encoded, theta, self.learning_rate)

            if i % self.save_history_every_n_iter == 0 or i == self.num_iterations - 1:
                self.history.append({
                    "iteration": i, 
                    "cost": cost, 
                    "thetas": theta.copy().flatten()
                })

        self.theta = theta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.to_numpy()
        m_samples = X.shape[0]
        # add bias term (intercept) to each sample as in linear regression
        X_b = np.c_[
            np.ones((m_samples, 1)), 
            X
        ]  
        # get probabilities using the feed forward step
        probabilities = self.feed_forward(X_b, self.theta)
        if len(self.classes) > 2:
            # for multi-class classification, we take the class with the highest probability
            return self.classes[np.argmax(probabilities, axis=1)]
        else:
            # for binary classification, we cut at the threshold
            return np.where(probabilities >= self.threshold, self.classes[1], self.classes[0]).reshape(-1)