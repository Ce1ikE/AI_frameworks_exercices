from lib.StdScalar import StdScaler
from lib.DataGatherer import DataGatherer
from lib.Reporter import Reporter

from lib.LogisticRegression import LogisticRegressionCustom
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from lib.KMeans import KMeansCustom

from lib.Search import Search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def logistic_regression_exercice__binomial():
    # ------------- Gather Data ------------- #
    X_train, X_test, y_train, y_test = DataGatherer.gather_datasets("breast_cancer")

    # ------------- Feature Scaling ------------- #
    scaler = StdScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------- Custom Logistic Regression ------------- #
    learning_rate = 0.01
    num_iterations = 15000
    lr_c = LogisticRegressionCustom(
        num_iterations=num_iterations, 
        learning_rate=learning_rate, 
        threshold=0.5,
    )
    lr_c.run_logistic_regression(X_train, y_train)

    # ------------- Sklearn Logistic Regression ------------- #
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)

    # - Is it already applied the Sigmoid function? 
    #   Yes, the predict method of sklearn's LogisticRegression class applies the sigmoid function internally
    # - What you need to do?
    #   Nothing, just use the predict method directly to get class labels
    # - Is it ready to compare with your target?
    #   Yes

    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/logistic_regression_binomial")
    reporter.save_final_results(lr_c, lr, X_test, y_test)


def logistic_regression_exercice__multinomial():
    # ------------- Gather Data ------------- #
    X_train, X_test, y_train, y_test = DataGatherer.gather_datasets("digits")

    # ------------- Feature Scaling ------------- #
    # ...

    # ------------- Custom Logistic Regression ------------- #
    learning_rate = 0.01
    num_iterations = 15000
    lr_c = LogisticRegressionCustom(
        num_iterations=num_iterations, 
        learning_rate=learning_rate, 
        threshold=0.5,
    )
    lr_c.run_logistic_regression(X_train, y_train)

    # ------------- Sklearn Logistic Regression ------------- #
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)

    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/logistic_regression_multinomial")
    reporter.save_final_results(lr_c, lr, X_test, y_test)

def logistic_regression_exercice__hyperparameter_tuning():
    # ------------- Gather Data ------------- #
    X_train, X_test, y_train, y_test = DataGatherer.gather_datasets("breast_cancer")

    # ------------- Feature Scaling ------------- #
    scaler = StdScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------- Custom Logistic Regression ------------- #
    Searcher = Search(LogisticRegressionCustom())
    param_grid = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "num_iterations": [5000, 10000, 15000]
    }
    best_model, best_score, best_params = Searcher.grid_search(X_train, y_train, X_test, y_test, param_grid)

    # ------------- Sklearn Logistic Regression ------------- #
    lr = LogisticRegression(max_iter=10000)
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
    grid_search = GridSearchCV(lr, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/logistic_regression_hyperparameter_tuning")
    reporter.save_final_results(best_model, grid_search.best_estimator_, X_test, y_test)

    # ------------- Gather Data ------------- #
    X_train, X_test, y_train, y_test = DataGatherer.gather_datasets("digits")
    # ------------- Feature Scaling ------------- #
    # ...

    # ------------- Custom Logistic Regression ------------- #
    Searcher = Search(LogisticRegressionCustom())
    param_grid = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "num_iterations": [5000, 10000, 15000]
    }
    best_model, best_score, best_params = Searcher.grid_search(X_train, y_train, X_test, y_test, param_grid)

    # ------------- Sklearn Logistic Regression ------------- #
    lr = LogisticRegression(max_iter=10000)
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
    grid_search = GridSearchCV(lr, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/logistic_regression_hyperparameter_tuning_digits")
    reporter.save_final_results(best_model, grid_search.best_estimator_, X_test, y_test)

def kmeans_exercice():
    n_clusters = 5
    m_samples = 1000
    random_state = 42
    # ------------- Gather Data ------------- #
    X_train, _, y_true, _ = DataGatherer.gather_datasets("blobs",n_clusters=n_clusters, m_samples=m_samples)

    # ------------- Feature Scaling ------------- #
    scaler = StdScaler()
    X_train = scaler.fit_transform(X_train)

    # ------------- Custom KMeans ------------- #
    kmeans_c = KMeansCustom(n_clusters=n_clusters, max_iter=300, random_state=random_state)
    kmeans_c.run_kmeans(X_train)

    # ------------- Sklearn KMeans ------------- #
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=random_state)
    kmeans.fit(X_train)

    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/kmeans")
    reporter.save_final_results_kmeans(X_train, y_true, kmeans_c, kmeans)


def kmeans_exercice_3D():
    n_clusters = 5
    m_samples = 1000
    random_state = 42
    # ------------- Gather Data ------------- #
    X_train, _, y_true, _ = DataGatherer.gather_datasets("blobs",n_clusters=n_clusters, m_samples=m_samples, random_state=random_state,n_features=3)

    # ------------- Feature Scaling ------------- #
    scaler = StdScaler()
    X_train = scaler.fit_transform(X_train)

    # ------------- Custom KMeans ------------- #
    kmeans_c = KMeansCustom(n_clusters=n_clusters, max_iter=300, random_state=random_state)
    kmeans_c.run_kmeans(X_train)

    # ------------- Sklearn KMeans ------------- #
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=random_state)
    kmeans.fit(X_train)

    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/kmeans_3D")
    reporter.save_final_results_kmeans_3D(X_train, y_true, kmeans_c, kmeans)

def kmeans_exercice__elbow():
    n_clusters_range = range(1, 15)
    m_samples = 1000
    random_state = 42
    # ------------- Gather Data ------------- #
    X_train, _, y_true, _ = DataGatherer.gather_datasets("blobs",n_clusters=5, m_samples=m_samples, random_state=random_state)

    # ------------- Feature Scaling ------------- #
    scaler = StdScaler()
    X_train = scaler.fit_transform(X_train)

    # ------------- Custom KMeans ------------- #
    inertias_c = []
    for n_clusters in n_clusters_range:
        kmeans_c = KMeansCustom(n_clusters=n_clusters, max_iter=300, random_state=random_state)
        kmeans_c.run_kmeans(X_train)
        inertias_c.append(kmeans_c.inertia)

    # ------------- Sklearn KMeans ------------- #
    inertias = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=random_state)
        kmeans.fit(X_train)
        inertias.append(kmeans.inertia_)

    # ------------- final results ------------- #
    reporter = Reporter(save_path="./results/kmeans_elbow")
    reporter.plot_elbow(n_clusters_range, inertias_c, inertias)


def main():
    logistic_regression_exercice__binomial()
    logistic_regression_exercice__multinomial()

    # Question 1: Which parameters would you use for the breast cancer dataset?
    # I would use:
    # - penalty='l2' (to avoid overfitting)
    # - max_iter=10000 (to ensure convergence)
    # Question 2: Perform a search.
    # Question 3: Check the model performance with the new parameters. Is there a difference?
    # Yes, the accuracy improved for both sklearn and my implementation to around 99.1%
    # Question 4:Check the Digits dataset, for this one which parameters would you use
    # Question 5: Perform a search.
    # Question 6: Check the model performance with the new parameters. Is there a difference?
    logistic_regression_exercice__hyperparameter_tuning()

    kmeans_exercice()
    kmeans_exercice_3D()
    kmeans_exercice__elbow()

if __name__ == "__main__":
    main()

    # Exercise:
    # Explain what do you understand from the pairplot (see kmeans jupyter notebook)
    # the plot shows (left below and right above) 2 features ploted against each other, showing the blobs data
    # where the colors represent the different clusters. The diagonal shows the distribution of each feature.
    # plot 1 : feature_1 
    # plot 2 : feature_2 vs feature_1
    # plot 3 : feature_1 vs feature_2
    # plot 4 : feature_2

    # And what about the parameters of the model?
    # Logistic Regression from Scikit-Learn has the following parameters:
    # - penalty='l2',
    # - dual=False,
    # - tol=0.0001,
    # - C=1.0,
    # - fit_intercept=True,
    # - intercept_scaling=1,
    # - class_weight=None,
    # - random_state=None,
    # - solver='lbfgs',
    # - max_iter=100,
    # - multi_class='auto',
    # - verbose=0,
    # - warm_start=False,
    # - n_jobs=None,
    # - l1_ratio=None
    # All the parameters are followed by the default attribute from the model.
    # Do you know what they mean?
    # No not exactly in detail, only some of them, like :
    # - fit_intercept: whether to include an intercept term in the model (bias in my implementation)
    # - max_iter: maximum number of iterations (15000 in my implementation)
    # - solver: algorithm to use in the optimization problem (gradient descent in my implementation see my comments in my logistic regression code)
    # - random_state: seed used by the random number generator (i do not use it in my implementation)
    # - C: I think it is related to regularization, which I do not use in my implementation. SVM also has a C parameter
    # - penalty: I think it is also related to regularization (L1, L2 are the ones I know), which I do not use in my implementation
    # - multi_class: I think it is related to how to handle multi-class classification
    # as far as I know regularization is to keep the weights small to avoid overfitting
    # something you add to the cost function to penalize large weights (I think that Scikit has Ridge and Lasso regression for that)
    # - tol: is just a threshold to stop the optimization when the change in cost function is below this value (same as in LR)