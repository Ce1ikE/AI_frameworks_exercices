import matplotlib.pyplot as plt
import time
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from .LinearRegression import LinearRegressionCustom

# for plot customizations i recommend:
# https://python-graph-gallery.com/

class Reporter:
    def __init__(
        self, 
        label_x: list[str]= ['Population of City in 10,000s'], 
        label_y: str='Profit in $10,000s',
        save_path=f'./results/linear_regression',
        fontdict=None
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
        self.label_x = label_x
        self.label_y = label_y
        self.fontdict = fontdict or {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def to_serializable(self,obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self.to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.to_serializable(v) for v in obj]
        return obj

    def plot_result(self, lr_c_history):
        iterations = [r['iteration'] for r in lr_c_history]
        costs = [r['cost'] for r in lr_c_history]
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

    def plot_linear_regression_over_history(self, lc: LinearRegressionCustom,X, y,):
        for i, record in enumerate(lc.history):
            if i == 0:
                i = 'initial'
            if len(record['thetas']) == 2:
                self.dimension = 2
                self.plot_2d(lc.X_b, y, record['thetas'], i)
            elif len(record['thetas']) == 3:
                self.dimension = 3
                self.plot_3d(lc.X_b, y, record['thetas'], i)

    def plot_3d(self, X, y, theta, i):
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

    def plot_2d(self, X, y, theta, i):
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
            
    def save_result(self, lr_c: LinearRegressionCustom):
        with open(self.save_path / "linear_regression_results.json", 'w') as f:
            json.dump(self.to_serializable(lr_c.history), f, indent=4)

    def save_final_results(self, lr_c: LinearRegressionCustom, lr: LinearRegression):
        final_results = {
            'custom': {
                "learning_rate": lr_c.learning_rate,
                "num_iterations": lr_c.num_iterations,
                "final_cost": lr_c.history[-1]["cost"],
                "theta": lr_c.history[-1]["thetas"],
            },
            'sklearn': {
                'intercept': lr.intercept_.tolist(),
                'coefficient': lr.coef_.tolist(),
            }
        }

        with open(self.save_path / 'final_results.json', 'w') as f:
            json.dump(self.to_serializable(final_results), f, indent=4)