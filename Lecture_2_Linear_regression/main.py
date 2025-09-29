import numpy as np
import pandas as pd
from statistics import linear_regression 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lib.LinearRegression import LinearRegressionCustom
import json
from pathlib import Path

def plot_data(X, y):
    plt.scatter(X, y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Food Truck Profit vs City Population')
    plt.show()

def read_data(path: Path = './datasets/food_truck.txt'):
    #read from dataset
    data = pd.read_csv(path, header = None, delimiter = ",") 
    # read first column, will be put in a 'series' variable
    X = data.iloc[:,0] 
    print('X.shape: ', X.shape)
    # read second column, will be put in a 'series' variable
    y = data.iloc[:,1] 
    print('y.shape: ', y.shape)
    # number of training examples
    m = len(y) 
    print('Number of samples:', m)
    # view first few rows of the data
    print(data.head()) 
    plot_data(X, y)
    return X, y

def linear_regression_exercice():
    X, y = read_data("./datasets/food_truck.txt")
    
    # ------------- Custom Linear Regression ------------- #
    lr_c = LinearRegressionCustom(
        save_plots=True, 
        save_results=True, 
        num_iterations=100000, 
        learning_rate=0.01
    )
    lr_c.run_linear_regression(X, y)
    
    # ------------- Sklearn Regression ------------- #
    lr = LinearRegression().fit(
        X=X.to_numpy()[:,np.newaxis], 
        y=y.to_numpy()[:,np.newaxis]
    )

    # ------------- final results ------------- #
    final_results = {
        'custom': lr_c.result[-1] if lr_c.result else None,
        'sklearn': {
            'intercept': lr.intercept_[0],
            'coefficient': lr.coef_[0][0]
        }
    }
    with open(lr_c.save_path / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)


def linear_regression_multivariate_exercice():
    pass

def main():
    linear_regression_exercice()
    linear_regression_multivariate_exercice()

if __name__ == "__main__":
    main()