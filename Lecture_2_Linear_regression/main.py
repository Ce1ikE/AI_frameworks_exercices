import numpy as np
import pandas as pd
from statistics import linear_regression 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lib.LinearRegression import LinearRegressionCustom
from lib.DataGatherer import DataGatherer
from lib.StdScaler import StdScaler
import json
from pathlib import Path

def linear_regression_exercice():
    # ------------- Gather Data ------------- #
    #  - 1st column : the population of a city and 
    #  - 2nd column : the profit of a food truck in that city. (A negative value for profit indicates a loss)
    path = Path(__file__).parent / 'datasets/food_truck.txt'
    X, y = DataGatherer(plot=True, path=path).read_data(X_cols=[0], y_col=1)

    # ------------- Feature Scaling ------------- #
    # ...

    # ------------- Custom Linear Regression ------------- #
    lr_c = LinearRegressionCustom(
        save_plots=True, 
        save_results=True, 
        num_iterations=100000, 
        learning_rate=0.01,
        check_costs_every_n_iter=5000,

    )
    lr_c.run_linear_regression(
        X=X, y=y,
        label_x=['Population of City in 10,000s'],
        label_y='Profit in $10,000s'
    )

    # ------------- Sklearn Regression ------------- #
    lr = LinearRegression().fit(
        X=X.to_numpy(), 
        y=y.to_numpy()
    )

    # ------------- final results ------------- #
    final_results = {
        'custom': lr_c.result[-1] if lr_c.result else None,
        'sklearn': {
            'intercept': lr.intercept_.tolist(),
            'coefficient': lr.coef_.tolist(),
        }
    }
    with open(lr_c.save_path / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)


def linear_regression_multivariate_exercice():
    # ------------- Gather Data ------------- #
    # - 1st column : size of the house (in square feet)
    # - 2nd column : the number of bedrooms
    # - 3rd column : the price of the house
    path = Path(__file__).parent / 'datasets/housing_prices.txt'
    X, y = DataGatherer(plot=True, path=path).read_data(X_cols=[0,1], y_col=2)

    # ------------- Feature Scaling ------------- #
    # experienced overflow errors when not scaling the features 
    # so well here is my StdScaler (like sklearn's StandardScaler)
    scaler = StdScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    # ------------- Custom Linear Regression ------------- #
    lr_c = LinearRegressionCustom(
        save_plots=True, 
        save_results=True, 
        num_iterations=100000, 
        learning_rate=0.01,
        check_costs_every_n_iter=5000,
    )
    lr_c.run_linear_regression(
        X=X, y=y,
        label_x=['Size of the house (in square feet)', 'Number of bedrooms'],
        label_y='Price of the house'
    )

    # ------------- Sklearn Regression ------------- #
    lr = LinearRegression().fit(
        X=X.to_numpy(), 
        y=y.to_numpy()
    )
    # ------------- final results ------------- #
    final_results = {
        'custom': lr_c.result[-1] if lr_c.result else None,
        'sklearn': {
            'intercept': lr.intercept_.tolist(),
            'coefficient': lr.coef_.tolist(),
        }
    }
    with open(lr_c.save_path / 'final_results_multivariate.json', 'w') as f:
        json.dump(final_results, f, indent=4)

def main():
    linear_regression_exercice()
    linear_regression_multivariate_exercice()

if __name__ == "__main__":
    main()