import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class DataGatherer:
    def __init__(self,plot = False,path: Path = './datasets/food_truck.txt'):
        self.path = path
        self.plot = plot

    def read_data(self, X_cols: list = [0], y_col: int = 1):
        if X_cols is None or len(X_cols) == 0:
            raise ValueError("X_cols must be a list with at least one column index.")
        if y_col is None:
            raise ValueError("y_col must be a valid column index.")
        if X_cols == [y_col] or y_col in X_cols:
            raise ValueError("X_cols and y_col must refer to different columns.")

        #read from dataset
        data = pd.read_csv(self.path, header = None, delimiter = ",") 
        # read first column, will be put in a 'series' variable
        X = data.iloc[:,X_cols] 
        print('X.shape: ', X.shape)
        # read second column, will be put in a 'series' variable
        y = data.iloc[:,y_col] 
        print('y.shape: ', y.shape)
        # number of training examples
        m = len(y) 
        print('Number of samples:', m)
        # view first few rows of the data
        print(data.head()) 

        if self.plot:
            self.plot_data(X, y)

        return X, y

    def plot_data(self, X: pd.DataFrame, y: pd.Series):
        # if plot is 2D else 3D
        if X.shape[1] == 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, color='blue', label='Data points')
            plt.title('Data Points')
            plt.xlabel('X')
        else:
            # https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X[0], X[1], y, color='blue', label='Data points')
            ax.set_title('Data Points')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.savefig(self.path.parent / f'{self.path.stem}.svg', format='svg')
        plt.close()