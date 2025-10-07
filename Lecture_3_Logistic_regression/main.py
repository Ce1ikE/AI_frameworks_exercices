from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from lib.LogisticRegression import LogisticRegressionCustom
from lib.StdScalar import StdScaler
from lib.DataGatherer import DataGatherer
from lib.Reporter import Reporter
from lib.KMeans import KMeansCustom


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
    reporter = Reporter()
    reporter.plot_result(lr_c.history)
    reporter.save_result(lr_c)
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
    reporter = Reporter()
    reporter.plot_result(lr_c.history)
    reporter.save_result(lr_c)
    reporter.save_final_results(lr_c, lr, X_test, y_test)


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
    kmeans_exercice()
    kmeans_exercice_3D()
    kmeans_exercice__elbow()

    # Exercise:
    # Explain what do you understand from the pairplot (see kmeans jupyter notebook)
    # the plot shows (left below and right above) 2 features ploted against each other, showing the blobs data
    # where the colors represent the different clusters. The diagonal shows the distribution of each feature.
    # plot 1 : feature_1 
    # plot 2 : feature_2 vs feature_1
    # plot 3 : feature_1 vs feature_2
    # plot 4 : feature_2

    

if __name__ == "__main__":
    main()
