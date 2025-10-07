from sklearn.datasets import load_breast_cancer , load_digits, make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd

class DataGatherer:
    def __init__(self):
        pass

    @staticmethod
    def gather_datasets(dataset_name: str, **kwargs):
        if dataset_name == "breast_cancer":
            X, y = load_breast_cancer(return_X_y=True,as_frame=True)
            y: pd.Series
            X: pd.DataFrame
            X_TEST_SIZE = 0.2
            RANDOM_STATE = 42
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=X_TEST_SIZE, random_state=RANDOM_STATE
            )

            return X_train, X_test, y_train, y_test
        elif dataset_name == "digits":
            X, y = load_digits(return_X_y=True, as_frame=True)
            X: pd.DataFrame
            y: pd.Series
            X_TEST_SIZE = 0.2
            RANDOM_STATE = 42
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=X_TEST_SIZE, random_state=RANDOM_STATE
            )
            return X_train, X_test, y_train, y_test
        
        elif dataset_name == "blobs":
            m_samples = kwargs.get("m_samples", 1000)
            n_clusters = kwargs.get("n_clusters", 5)
            RANDOM_STATE = kwargs.get("random_state", 42)
            n_features = kwargs.get("n_features", 2)
            X, y = make_blobs(n_samples=m_samples, centers=n_clusters, n_features=n_features, random_state=RANDOM_STATE)
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
            y = pd.Series(y)
            return X, None, y, None
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")