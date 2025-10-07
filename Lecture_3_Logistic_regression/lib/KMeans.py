import numpy as np
import pandas as pd

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None, distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.distance_metric = distance_metric
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)

    def euclidean_distance(self, a, b):
        return np.linalg.norm(a - b, axis=-1)
    
    def cosine_distance(self, a, b):
        # 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # https://www.geeksforgeeks.org/python/how-to-calculate-cosine-similarity-in-python/
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        cosine_similarity = np.dot(a_norm, b_norm.T)
        distances = 1 - cosine_similarity

        return distances
    
    def _calculate_distances(self, X):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(X[:, np.newaxis], self.centroids)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(X[:, np.newaxis], self.centroids)
        else:
            raise ValueError(f"Distance metric '{self.distance_metric}' is not supported.")

    def _calculate_inertia(self, distances, n_samples=None):
        # https://medium.com/@naomirejoice4/key-differences-among-inertia-distortion-and-silhouette-score-in-clustering-4dbb454f2fd1
        # https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/
        # compute inertia:
        # "WCSS measures how well the data points are clustered around their respective centroids. 
        # It is defined as the sum of the squared distances between each point and its cluster centroid"
        self.inertia = np.sum(
            np.square(
                distances[np.arange(n_samples), self.labels]
            )
        )

    # https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670/
    # https://medium.com/@avijit.bhattacharjee1996/implementing-k-means-clustering-from-scratch-in-python-a277c23563ac
    def run_kmeans(self, X: pd.DataFrame):
        n_samples, n_features = X.shape
        X = X.to_numpy()

        # steps:
        # 1) init centroids randomly across the dataset
        # 2) assign each point to the nearest centroid
        # 3) recompute centroids as the mean of the assigned points
        # 4) repeat steps 2 and 3 until convergence or max_iter
        # -> convergence is reached when centroids do not change significantly

        # init centroids
        random_indices = self.rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        distances = None
        for i in range(self.max_iter):
            # compute distances from points to centroids
            distances = self._calculate_distances(X)
            
            # assign labels based on closest centroid
            self.labels = np.argmin(distances, axis=1)
            # recompute centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.allclose(new_centroids,self.centroids, atol=self.tol, rtol=0):
                break

            self.centroids = new_centroids

        self._calculate_inertia(distances, n_samples=n_samples)

    def predict(self, X: pd.DataFrame):
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        X = X.to_numpy() if hasattr(X, 'to_numpy') else X
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    