from pathlib import Path
from sklearn.linear_model import LinearRegression
from lib.LinearRegression import LinearRegressionCustom
from lib.DataGatherer import DataGatherer
from lib.StdScaler import StdScaler
from lib.Reporter import Reporter

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
        num_iterations=100000, 
        learning_rate=0.01,
        save_history_every_n_iter=5000,
    )
    lr_c.run_linear_regression(X=X, y=y)

    # ------------- Sklearn Regression ------------- #
    lr = LinearRegression().fit(
        X=X.to_numpy(), 
        y=y.to_numpy()
    )

    # ------------- final results ------------- #
    reporter = Reporter(
        label_x=['Population of City in 10,000s'],
        label_y='Profit in $10,000s'
    )
    reporter.plot_result(lr_c.history)
    reporter.save_result(lr_c)
    reporter.save_final_results(lr_c, lr)
    reporter.plot_linear_regression_over_history(lr_c, X, y) 


 


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
        num_iterations=100000, 
        learning_rate=0.01,
        save_history_every_n_iter=5000,
    )
    lr_c.run_linear_regression(X=X, y=y)

    # ------------- Sklearn Regression ------------- #
    lr = LinearRegression().fit(
        X=X.to_numpy(), 
        y=y.to_numpy()
    )
    # ------------- final results ------------- #
    reporter = Reporter(
        label_x=['Size of House (standardized)', 'Number of Bedrooms (standardized)'],
        label_y='Price of House (standardized)'
    )
    reporter.plot_result(lr_c.history)
    reporter.save_result(lr_c)
    reporter.save_final_results(lr_c, lr)
    reporter.plot_linear_regression_over_history(lr_c, X, y)

def main():
    linear_regression_exercice()
    linear_regression_multivariate_exercice()

if __name__ == "__main__":
    main()