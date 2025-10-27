
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from .DataHandler import DataHandler

class Reporter:
    def __init__(
        self,
        save_path: str | Path = "results",
        fontdict: dict | None = None,
    ):
        self.fontdict = fontdict or {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }
        
        if isinstance(save_path, str):
            self.save_path = Path(save_path)
        else:
            self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)      
        self.save_path = self.save_path / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path.mkdir(parents=True, exist_ok=True)  

        # to reduce clutter, we create subfolders for different types of reports
        self.subfolders = {
            "analysis": self.save_path / "analysis",
            "kmeans": self.save_path / "kmeans",
            "dbscan": self.save_path / "dbscan",
            "optics": self.save_path / "optics",
            "agglomerative": self.save_path / "agglomerative",
            "maps": self.save_path / "maps"
        }
        for f in self.subfolders.values():
            f.mkdir(parents=True, exist_ok=True)

    def plot__dataset_info(self, df: pd.DataFrame, title: str = ""):
        col = list(df.columns)
        col.remove('country')
        self.categorical_features = ['country']
        self.numerical_features = [*col] 

        print("\nCategorical Features:", self.categorical_features)
        print("Numerical Features:", self.numerical_features)

        plt.suptitle("Distribution of Numerical Features", y=1.02, **self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        df[self.numerical_features].hist(bins=15, figsize=(15, 10))
        plt.savefig(self.save_path / f'{title}__numerical_features_distribution.svg', format='svg')
        plt.close()

    def plot__distribution_compressed_data(self, df: pd.DataFrame, title: str = "Compressed Data Distribution"):
        
        for col in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], bins=15, kde=True)
            plt.title(f"Distribution of {col}", fontdict=self.fontdict)
            plt.xlabel(col, fontdict=self.fontdict)
            plt.ylabel("Frequency", fontdict=self.fontdict)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(self.subfolders["analysis"] / f'{title.replace(" ", "_").lower()}_{col}_distribution.svg', format='svg')
            plt.close()

    def plot__features(self, df: pd.DataFrame, features: list[str], title: str = ""):
        # we plot 3 barplots for each feature: top 5 countries, medium 5 countries, low 5 countries
        # [top 5 countries, medium 5 countries, low 5 countries] * number of features
        batch_size = 3
        for i in range(0, len(features), batch_size):
            subset = features[i:i+batch_size]
            fig_size_height = 10 * len(subset)
            col_size_width = 30
            fig = plt.subplots(nrows = len(subset),ncols = 3,figsize = (col_size_width,fig_size_height))
            for n, feature in enumerate(subset):
                for m, plt_title in enumerate(['Top 5 Countries', 'Medium 5 Countries', 'Low 5 Countries']):
                    plt.subplot(len(subset),3, n*3 + m + 1)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.title(plt_title, fontdict=self.fontdict)
                    plt.xlabel("Country")
                    plt.ylabel(feature)
                    if plt_title == 'Top 5 Countries':
                        start_idx = 0
                        idx_range = 5
                    elif plt_title == 'Medium 5 Countries':
                        start_idx = len(df)//2 - 2
                        idx_range = 5
                    elif plt_title == 'Low 5 Countries':
                        start_idx = len(df) - 5
                        idx_range = 5

                    sorted_df = df.sort_values(by=feature, ascending=False).iloc[start_idx:start_idx+idx_range]
                    sns.barplot(
                        data=sorted_df,
                        x='country',
                        y=feature,
                        ax=plt.gca(),
                        hue='country',
                        legend=False,
                        edgecolor='black',
                        linewidth=2,
                        # palette="Blues"
                    )
                    plt.tight_layout()
                    for rect in plt.gca().patches:
                        plt.gca().text(
                            rect.get_x() + rect.get_width()/2, 
                            rect.get_height(), 
                            round(rect.get_height(),2), 
                            horizontalalignment='center', 
                            fontdict=self.fontdict
                        )
                        
            plt.suptitle(f"{title}__Features_Analysis", y=1.02, **self.fontdict)
            plt.savefig(self.subfolders["analysis"] / f'{title}__features_analysis_{i//batch_size + 1}.svg', format='svg')
            plt.close()

    def plot__correlation_matrix(self, df: pd.DataFrame, title: str = "Correlation Matrix of Numerical Features"):
        # - Looking to each feature and the respective values per country,
        # can you make some assumptions between features? Can you see some correlation?
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[self.numerical_features].corr(method='pearson')
        annotations = np.where(
            np.abs(correlation_matrix) > 0.5, 
            correlation_matrix.round(2).astype(dtype=str).copy(deep=True), 
            ""
        )
        sns.heatmap(
            correlation_matrix, 
            annot=annotations,
            annot_kws=self.fontdict, 
            cmap=plt.get_cmap(name="RdBu"), 
            cbar=True,
            fmt="s",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            shading='auto',
            linecolor='gray',
        )
        plt.title(title, fontdict=self.fontdict)
        plt.savefig(self.subfolders["analysis"] / f'correlation_matrix.svg', format='svg')
        plt.close()
        # we can deduce some correlations from the heatmap:
        # - income and gdpp are highly positively correlated.
        # - child_mort and life_expec are highly negatively correlated.
        # - child_mort and income are also negatively correlated.

    def plot__kmeans_evaluation_metrics(self, k_range: range, silhouette_scores: list, inertias: list, title: str = "KMeans Clustering Evaluation Metrics"):
        plt.figure(figsize=(14, 6))
        plt.suptitle(title, y=1.02, **self.fontdict)
        ax = plt.subplot(1, 2, 1)
        ax.plot(k_range, silhouette_scores, marker='o')
        ax.set_title("Silhouette Scores for KMeans Clustering", fontdict=self.fontdict)
        ax.set_xlabel("Number of clusters (k)", fontdict=self.fontdict)
        ax.set_ylabel("Silhouette Score", fontdict=self.fontdict)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Plotting inertia
        ax = plt.subplot(1, 2, 2)
        ax.plot(k_range, inertias, marker='o')
        ax.set_title("Inertia for KMeans Clustering", fontdict=self.fontdict)
        ax.set_xlabel("Number of clusters (k)", fontdict=self.fontdict)
        ax.set_ylabel("Inertia", fontdict=self.fontdict)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.subfolders["analysis"] / f'kmeans_evaluation_metrics.svg', format='svg')
        plt.close()

    def plot__3d(self, X: np.ndarray, cluster_labels: np.ndarray, algo_name: str, title: str = ""):
        import plotly.express as px
        fig = px.scatter_3d(
            x=X[:, 0], 
            y=X[:, 1], 
            z=X[:, 2],
            color=cluster_labels.astype(str),
            title=title,
            labels={'x':'Feature 1', 'y':'Feature 2', 'z':'Feature 3'}
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
        title = title.replace(" ", "_")
        fig.write_html(f'{self.subfolders[algo_name.lower()]}/{title}_{algo_name}_3d_visualization.html')

        return fig

    def plot__k_distance_graph(self,n_neighbors:int, distances: np.ndarray, title: str = "k-Distance Graph"):
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.legend([f"{n_neighbors}th Nearest Neighbor Distance" for n_neighbors in range(1, n_neighbors + 1)])
        plt.title(title, fontdict=self.fontdict)
        plt.xlabel("Data Points sorted by distance")
        plt.ylabel(f"Distance to Kth Nearest Neighbor")
        plt.grid(True, linestyle='--', alpha=0.7)
        title = title.replace(" ", "_")
        plt.savefig(self.subfolders["analysis"] / f'{title}_k_distance_graph.svg', format='svg')
        plt.close()


    def plot__clusters(self, X: np.ndarray, labels: np.ndarray, title: str, algo_name: str):
        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        colors = sns.color_palette("hsv", len(unique_labels))
        color_map = {label: color for label, color in zip(unique_labels, colors)}
        color_map[-1] = (0.6, 0.6, 0.6)  # gray for noise

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X[:, 0], 
            X[:, 1],
            c=[color_map[label] for label in labels],
            alpha=0.7, 
            edgecolor='black'
        )
        plt.title(f"{algo_name} Clustering ({n_clusters} clusters)", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(self.subfolders[algo_name.lower()] / f'{title}_{algo_name}_clusters.svg', format='svg')
        plt.close()

    def plot__world_map_clusters(self, cluster_data: pd.DataFrame, cluster_labels: np.ndarray, title: str = "World Map Clustering"):
        import plotly.express as px
        cluster_data = cluster_data.copy()
        cluster_data['Cluster'] = cluster_labels.astype(str)
        # in the dataframe, country names are in the index
        if 'country' not in cluster_data.columns:
            cluster_data.reset_index(inplace=True)
            cluster_data.rename(columns={"index": "country"}, inplace=True)
                
        #  ['country','Health', 'Trade', 'Finance', 'Cluster'] are the columns then
        # depending on the country names, plotly will color the countries based on the cluster they belong to
        fig = px.choropleth(
            cluster_data,
            locations="country",
            locationmode="country names",
            color="Cluster",
            title=title,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        title = title.replace(" ", "_")
        fig.write_html(f'{self.subfolders["maps"]}/{title}_world_map_clustering.html')

        return fig
