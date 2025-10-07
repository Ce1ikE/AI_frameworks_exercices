from matplotlib import pyplot as plt
import time
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from .LogisticRegression import LogisticRegressionCustom
from sklearn.metrics import confusion_matrix , silhouette_score
from sklearn.cluster import KMeans
from .KMeans import KMeansCustom
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px


class Reporter:
    def __init__(
        self, 
        label_x: list[str]= ['Population of City in 10,000s'], 
        label_y: str='Profit in $10,000s',
        save_path=f'./results/logistic_regression',
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
    
    # ----------------- Logistic Regression ----------------- #
    
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
        plt.savefig(self.save_path / 'logistic_regression_cost.svg', format='svg')
        plt.close()

    def save_result(self, lr_c: LogisticRegressionCustom):
        with open(self.save_path / "logistic_regression_results.json", 'w') as f:
            json.dump(self.to_serializable(lr_c.history), f, indent=4)

    def save_final_results(
        self, 
        lr_c: LogisticRegressionCustom, 
        lr: LogisticRegression,
        X_test,
        y_test,
    ):
        # --- custom logistic regression --- #
        y_pred = lr_c.predict(X_test)
        accuracy_lr_c = (y_pred == y_test).mean()
        class_labels = np.unique(np.concatenate((y_test, y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        self.save_confusion_matrix(cm, class_labels, name='custom')

        # --- sklearn logistic regression --- # 
        y_pred = lr.predict(X_test)
        accuracy_lr = (y_pred == y_test).mean()
        class_labels = np.unique(np.concatenate((y_test, y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        self.save_confusion_matrix(cm, class_labels, name='sklearn')

        final_results = {
            "custom": {
                "accuracy": accuracy_lr_c,
                "thetas": lr_c.history[-1]['thetas'] if lr_c.history else None,
                "final_cost": lr_c.history[-1]["cost"] if lr_c.history else None,
                "learning_rate": lr_c.learning_rate,
                "num_iterations": lr_c.num_iterations,
            },
            "sklearn": {
                "accuracy": accuracy_lr,
                "intercept": lr.intercept_.tolist(),
                "coefficient": lr.coef_.tolist(),
            }
        }

        with open(self.save_path / 'final_results.json', 'w') as f:
            json.dump(self.to_serializable(final_results), f, indent=4)
    
    def save_confusion_matrix(self, cm, class_labels,name: str):
        annotations = [[str(cm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        plt.figure("Confusion Matrix")
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, annotations[i][j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.savefig(self.save_path / f"confusion_matrix_{name}.svg", format='svg')
        plt.close()

    # ----------------- KMeans ----------------- #

    def save_final_results_kmeans(
        self, 
        X: pd.DataFrame,
        y_true,
        kmeans_c: KMeansCustom,
        kmeans: KMeans,
    ):
        class_labels = np.unique(np.concatenate((y_true, kmeans_c.labels, kmeans.labels_)))

        # --- custom kmeans --- #
        cm = confusion_matrix(y_true, kmeans_c.labels, labels=class_labels)
        self.save_confusion_matrix(cm, class_labels, name='custom_kmeans')
        silhouette_custom = silhouette_score(X, kmeans_c.labels) if len(class_labels) > 1 else None
        self.plot_kmeans(
            X=X.to_numpy() if hasattr(X, 'to_numpy') else X,
            y=kmeans_c.labels if kmeans_c.labels is not None else np.array([]),
            labels=class_labels,
            centroids=kmeans_c.centroids,
            title="Custom KMeans Clustering",
            name="custom_kmeans_clustering"
        )
        # --- sklearn kmeans --- # 
        cm = confusion_matrix(y_true, kmeans.labels_, labels=class_labels)
        self.save_confusion_matrix(cm, class_labels, name='sklearn_kmeans')
        silhouette_sklearn = silhouette_score(X, kmeans.labels_) if len(class_labels) > 1 else None
        self.plot_kmeans(
            X=X.to_numpy() if hasattr(X, 'to_numpy') else X,
            y=kmeans.labels_,
            labels=class_labels,
            centroids=kmeans.cluster_centers_,
            title="Sklearn KMeans Clustering",
            name="sklearn_kmeans_clustering"
        )
        final_results = {
            "custom_kmeans": {
                "clusters": kmeans_c.centroids.tolist() if kmeans_c.centroids is not None else None,
                "inertia": kmeans_c.inertia,
                "silhouette_score": silhouette_custom,
            },
            "sklearn_kmeans": {
                "clusters": kmeans.cluster_centers_.tolist(),
                "inertia": kmeans.inertia_,
                "silhouette_score": silhouette_sklearn,
            }
        }

        with open(self.save_path / 'final_results_kmeans.json', 'w') as f:
            json.dump(self.to_serializable(final_results), f, indent=4)

    def plot_kmeans(self, X, y, labels, centroids=None, title="KMeans Clustering", name: str="kmeans_clustering"):
        plt.figure(title)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', s=30, edgecolor='k')
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
            plt.legend()
        plt.title(title)
        plt.xlabel(labels[0], fontdict=self.fontdict)
        plt.ylabel(labels[1], fontdict=self.fontdict)
        plt.grid(
            visible=True, 
            which='both', 
            axis='both', 
            color='0.9', 
            linestyle='--', 
            linewidth=1,
            alpha=0.5
        )
        plt.savefig(self.save_path / f"{name}.svg", format='svg')
        plt.close()


    def plot_elbow(self, n_clusters_range, inertias_c, inertias):
        plt.figure("Elbow Method for Optimal k")
        plt.plot(n_clusters_range, inertias_c, marker='o', label='Custom KMeans', color='blue')
        plt.plot(n_clusters_range, inertias, marker='o', label='Sklearn KMeans', color='orange')
        plt.title(
            label='Elbow Method for Optimal k',
            fontdict=self.fontdict,
            pad=10,
            loc="left",
        )
        plt.xlabel('Number of Clusters (k)', fontdict=self.fontdict)
        plt.ylabel('Inertia', fontdict=self.fontdict)
        plt.xticks(n_clusters_range)
        plt.grid(
            visible=True, 
            which='both', 
            axis='both', 
            color='0.9', 
            linestyle='--', 
            linewidth=1,
            alpha=0.5
        )
        plt.legend()
        plt.savefig(self.save_path / 'elbow_method.svg', format='svg')
        plt.close()


    def save_final_results_kmeans_3D(
        self, 
        X: pd.DataFrame,
        y_true,
        kmeans_c: KMeansCustom,
        kmeans: KMeans,
    ):
        class_labels = np.unique(np.concatenate((y_true, kmeans_c.labels, kmeans.labels_)))

        # --- custom kmeans --- #
        cm = confusion_matrix(y_true, kmeans_c.labels, labels=class_labels)
        self.save_confusion_matrix(cm, class_labels, name='custom_kmeans_3D')
        silhouette_custom = silhouette_score(X, kmeans_c.labels) if len(class_labels) > 1 else None
        self.plot_kmeans_3D(
            X=X.to_numpy() if hasattr(X, 'to_numpy') else X,
            y=kmeans_c.labels if kmeans_c.labels is not None else np.array([]),
            labels=class_labels,
            centroids=kmeans_c.centroids,
            title="Custom KMeans 3D Clustering",
            name="custom_kmeans_3D_clustering"
        )
        # --- sklearn kmeans --- # 
        cm = confusion_matrix(y_true, kmeans.labels_, labels=class_labels)
        self.save_confusion_matrix(cm, class_labels, name='sklearn_kmeans_3D')
        silhouette_sklearn = silhouette_score(X, kmeans.labels_) if len(class_labels) > 1 else None
        self.plot_kmeans_3D(
            X=X.to_numpy() if hasattr(X, 'to_numpy') else X,
            y=kmeans.labels_,
            labels=class_labels,
            centroids=kmeans.cluster_centers_,
            title="Sklearn KMeans 3D Clustering",
            name="sklearn_kmeans_3D_clustering"
        )
        final_results = {
            "custom_kmeans_3D": {
                "clusters": kmeans_c.centroids.tolist() if kmeans_c.centroids is not None else None,
                "inertia": kmeans_c.inertia,
                "silhouette_score": silhouette_custom,
            },
            "sklearn_kmeans_3D": {
                "clusters": kmeans.cluster_centers_.tolist(),
                "inertia": kmeans.inertia_,
                "silhouette_score": silhouette_sklearn,
            }
        }

        with open(self.save_path / 'final_results_kmeans_3D.json', 'w') as f:
            json.dump(self.to_serializable(final_results),f, indent=4)

    def plot_kmeans_3D(self, X, y, labels, centroids=None, title="KMeans 3D Clustering", name: str="kmeans_3D_clustering"):
        fig = plt.figure(title)
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o', s=30, edgecolor='k')
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel(labels[0], fontdict=self.fontdict)
        ax.set_ylabel(labels[1], fontdict=self.fontdict)
        ax.set_zlabel(labels[2], fontdict=self.fontdict)
        plt.savefig(self.save_path / f"{name}.svg", format='svg')
        plt.close()

