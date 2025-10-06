import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
# for plot customizations i recommend:
# https://python-graph-gallery.com/

# combined multivariate and univariate linear regression into one class
# with some code that works for both univariate and multivariate linear regression
# plotting only works for univariate and bivariate linear regression (2 features + bias)
# which makes sense because well... it's hard to visualize more than 3 dimensions :)
class LinearRegressionCustom:
    def __init__(
        self,
        save_path=f'./results/linear_regression',
        save_plots: bool = True,
        save_results: bool = True,
        check_costs_every_n_iter: int = 1000,
        num_iterations: int = 100000,
        learning_rate: float = 0.01,
    ):
        self.save_path = Path(save_path + '_' +  time.strftime("%Y%m%d_%H%M%S"))
        try:
            self.save_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Directory {self.save_path} already exists.")
            # if it already exists, create a new directory with timestamp inside the existing one
            self.save_path = Path(save_path + '_' + time.strftime("%Y%m%d_%H%M%S") / str(int(time.time())))
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Results will be saved to {self.save_path}")

        # hyperparameters
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.result = []

        self.save_plots = save_plots
        self.save_results = save_results
        self.check_costs_every_n_iter = check_costs_every_n_iter

        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    # ------------------- Linear Regression ------------------- #

    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        # 1/2m * sum(h(x_i) - y_i)^2
        # where:
        #       h(x_i) = theta_0 + theta_1 * x_i
        # J == cost or error
        J = 1/(2*self.m) * np.sum(np.square(X.dot(theta) - y))
        return J

    def gradient_descent(self,X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float) -> np.ndarray:
        # theta_j = theta_j - learning_rate * d/dtheta_j(J(theta_0, theta_1))
        # where:
        #       J(theta_0, theta_1) = 1/2m * sum(h(x_i) - y_i)^2
        #       d/dtheta_j(J(theta_0, theta_1)) = 1/m * sum((h(x_i) - y_i) * x_i_j)

        # ***** DOES NOT WORK FOR MULTIVARIATE ***** #
        # so in the case of 
        #   f(x) = theta_0 + theta_1 * x this would be:
        # theta_0 = theta[0][0] - learning_rate * (1/m) * np.sum((X.dot(theta) - y)) 
        # theta_1 = theta[1][0] - learning_rate * (1/m) * np.sum((X.dot(theta) - y) * X[:,1][:,np.newaxis])
        # theta[0][0] = theta_0
        # theta[1][0] = theta_1
        # ***** DOES NOT WORK FOR MULTIVARIATE ***** #

        # however we can do this in a more general way 
        # for 
        #   f(x) = theta_0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn
        # 1) it's easier and faster to implement
        # 2) it works for both univariate and multivariate linear regression
        # 3) it's cleaner code
        gradient = (1/self.m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradient
        return theta
    
    def run_linear_regression(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        label_x: list[str] = ['Population of City in 10,000s'],
        label_y: str = 'Profit in $10,000s'
    ):
        self.label_x = label_x
        self.label_y = label_y
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1,1) # (m, 1)
        # number of samples , number of features
        self.m , self.n_features = X.shape[0], X.shape[1]
        self.dimension = self.n_features + 1 # +1 for bias term
        
        # ***** DOES NOT WORK FOR MULTIVARIATE ***** #
        # start off with a [0, 0] array
        # theta = np.zeros([2,1])
        # ***** DOES NOT WORK FOR MULTIVARIATE ***** #

        # we must have as many theta as there are features + 1 (for bias)
        # X.shape[1] is the number of features (m_samples, n_features)
        theta = np.zeros([self.dimension, 1])
        # https://numpy.org/doc/stable/reference/generated/numpy.c_.html
        # adds the prediction for y when x = 0 (bias term) (the x0 in f(x) = theta_0 * x0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn)
        # so it can be multiplied by theta
        # c_ is for column-wise concatenation
        X_b = np.c_[
            # array of [1, 1, 1, ..., 1] with shape (m, 1)
            np.ones((self.m, 1)),
            # expression to concatenate
            X
        ]     
        for i in range(self.num_iterations):
            theta = self.gradient_descent(X_b, y, theta, self.learning_rate)
            if i % self.check_costs_every_n_iter == 0:
                cost = self.compute_cost(X_b, y, theta)

                if self.save_results:
                    # ***** DOES NOT WORK FOR MULTIVARIATE ***** #
                    # self.result.append({'iteration': i, 'cost': cost, 'theta0': theta[0][0], 'theta1': theta[1][0]})
                    # print(f"Iteration {i}: Cost {cost}, Theta0: {theta[0][0]}, Theta1: {theta[1][0]}")
                    # ***** DOES NOT WORK FOR MULTIVARIATE ***** #
                    
                    self.result.append({
                        'iteration': i, 
                        'cost': cost, 
                        **{
                            f'theta{j}': theta[j][0] 
                            for j in range(theta.shape[0])
                        }
                    })
                    print(f"Iteration {i}: Cost {cost}, " + ', '.join(
                        [
                            f"Theta{j}: {theta[j][0]}" 
                            for j in range(theta.shape[0])
                        ]
                    ))

                if self.save_plots:
                    self.plot_linear_regression(X_b, y, theta,i)

        if self.save_results:
            self.plot_result()
            self.save_result()

    # ------------------- Results ------------------- #

    def plot_linear_regression(self, X, y, theta, i):
        if i == 0:
            i = 'initial'
        if self.dimension == 2:
            # 2D plot
            plt.figure(f"Linear Regression Fit")
            plt.title(
                label=f'Linear Regression Fit (Iteration {i})',
                fontdict=self.fontdict,
                pad=10,
                loc="left",
            )
            # stack in order to have grid behind the data points
            # 1) grid as background
            plt.grid(
                visible=True, 
                which='both', 
                axis='both', 
                color='0.9', 
                linestyle='--', 
                linewidth=1,
                alpha=0.5
            )
            # https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
            # 2) confidence interval as shaded area
            # some confidence interval
            x_sorted = np.argsort(X[:,1])
            x_vals = X[:,1][x_sorted]
            y_pred = np.dot(X, theta).ravel()
            ci = 1.96 * np.std(y - y_pred) / np.sqrt(len(X)) 
            upper_bound = y_pred + ci
            lower_bound = y_pred - ci
            plt.fill_between(
                x_vals, 
                upper_bound[x_sorted], 
                lower_bound[x_sorted], 
                color='red', 
                alpha=0.2, 
                label='95% Confidence Interval'
            )
            # 3) linear regression line
            plt.plot(X[:,1], np.dot(X, theta), color = 'red',linewidth=0.8)
            # 4) data points on top
            plt.scatter(X[:,1], y, marker='o', s=10, color='blue')
            
            plt.xlabel(self.label_x[0], fontdict=self.fontdict)
            plt.ylabel(self.label_y, fontdict=self.fontdict)
            plt.savefig(self.save_path / f'linear_regression_{i}.svg', format='svg')
            plt.close()
        elif self.dimension == 3:
            # 3D plot
            fig = plt.figure(f"Linear Regression Fit 3D")
            ax = fig.add_subplot(projection='3d')
            ax.set_title(
                f'Linear Regression Fit (Iteration {i})',
                fontdict=self.fontdict,
                pad=10,
                loc="left",
            )
            # 1) grid as background
            ax.grid(
                visible=True, 
                which='both', 
                axis='both', 
                color='0.9', 
                linestyle='--', 
                linewidth=1,
                alpha=0.5
            )
            # Create meshgrid for surface
            x1_range = np.linspace(X[:,1].min(), X[:,1].max(), 30)
            x2_range = np.linspace(X[:,2].min(), X[:,2].max(), 30)
            x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
            X_mesh = np.c_[np.ones(x1_mesh.ravel().shape), x1_mesh.ravel(), x2_mesh.ravel()]
            y_mesh = np.dot(X_mesh, theta).reshape(x1_mesh.shape)
            # 2) linear regression plane
            ax.plot_surface(
                x1_mesh, 
                x2_mesh, 
                y_mesh, 
                color='red', 
                alpha=0.5
            )
            # 3) data points on top
            ax.scatter(X[:,1], X[:,2], y.ravel(), marker='o', s=10, color='blue')

            ax.set_xlabel(self.label_x[0], fontdict=self.fontdict)
            ax.set_ylabel(self.label_x[1], fontdict=self.fontdict)
            ax.set_zlabel(self.label_y, fontdict=self.fontdict)
            plt.savefig(self.save_path / f'linear_regression_{i}_3d.svg', format='svg')
            plt.close()
        else:
            print("Plotting only supported for up to 3D (2 features + bias).")

    def plot_result(self):
        iterations = [r['iteration'] for r in self.result]
        costs = [r['cost'] for r in self.result]
        plt.figure("Cost Function over Iterations")
        plt.plot(iterations, costs, color = 'red',linewidth=0.8)
        plt.title(
            label='Cost Function over Iterations',
            fontdict=self.fontdict,
            pad=10,
            loc="left",
        )
        plt.grid(
            visible=True, 
            which='both', 
            axis='both', 
            color='0.9', 
            linestyle='--', 
            linewidth=1,
            alpha=0.5
        )
        plt.scatter(iterations, costs, marker='o', s=10, color='blue')
        plt.xlabel('Iterations', fontdict=self.fontdict)
        plt.ylabel('Cost', fontdict=self.fontdict)
        plt.savefig(self.save_path / 'linear_regression_cost.svg', format='svg')
        plt.close()

    def save_result(self):
        theta_finals = {
            f"final_theta{j}": self.result[-1][f'theta{j}'] 
            for j in range(len(self.result[-1])) if f'theta{j}' in self.result[-1]
        }

        self.result.append(
            {
                "theta": theta_finals,
                "final_cost": self.result[-1]["cost"],
                "learning_rate": self.learning_rate,
                "num_iterations": self.num_iterations
            }
        )
        with open(self.save_path / "linear_regression_results.json", 'w') as f:
            json.dump(self.result, f, indent=4)




