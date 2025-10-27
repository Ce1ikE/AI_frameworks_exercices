from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, KMeans, DBSCAN, AgglomerativeClustering 
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from lib.DataHandler import DataHandler
from lib.Reporter import Reporter

SAVE_DIR = Path('./results').absolute()
DATA_DIR = Path('./datasets/').absolute()
RANDOM_SEED = 42

reporter = Reporter(save_path=SAVE_DIR)

model_scores = {}

def countries_hands_on_analysis():
    # Load the dataset
    df = pd.read_csv(DATA_DIR / 'Country_data.csv')
    df = DataHandler.clean_data(df)
    df.to_csv(DATA_DIR / 'cleaned_Country_data.csv', index=True)
    reporter.plot__dataset_info(df, title="Country_Dataset_Info")
    
    features_to_plot = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer','gdpp']
    reporter.plot__features(df, features_to_plot,title="Country_Features_Distribution")
    reporter.plot__correlation_matrix(df)

    compressed_data = DataHandler.compress_country_data(df)
    compressed_data.to_csv(DATA_DIR / 'compressed_country_data.csv', index=True)

def countries_hands_on_training_kmeans(df: pd.DataFrame):
    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = range(2, 11)
    silhouette_scores = []
    inertias = []
    ### Elbow or Silhouette
    # - Explain with your words both methods and cite what are the differences.
    # the silhouette method measures how similar an point is to its own cluster compared to other clusters
    # the elbow method looks at the inertia (sum of squared distances to closest cluster center)
    # and finds the point where adding more clusters doesn't significantly reduce inertia
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)

    reporter.plot__kmeans_evaluation_metrics(k_range, silhouette_scores, inertias)
    
    max_silhouette = np.argmax(silhouette_scores)
    optimal_k_silhouette = k_range[max_silhouette]
    print(f"Optimal number of clusters (k) based on silhouette score: {optimal_k_silhouette}")
    kmeans = KMeans(n_clusters=optimal_k_silhouette, random_state=RANDOM_SEED)
    cluster_labels = kmeans.fit_predict(X_scaled)
    reporter.plot__clusters(
        X=X_scaled,
        labels=cluster_labels,
        title="KMeans Clustering (Silhouette Optimal k)",
        algo_name="KMeans",
    )
    reporter.plot__world_map_clusters(df, cluster_labels, title="World Map KMeans Clusters (Silhouette Optimal k)")

    # to determine optimal k based on inertia we look for the elbow point
    # normally this is done by plotting the inertia and looking for the point where the decrease in inertia starts to slow down
    # doing it during runtime can be done however by looking at the second derivative of the inertia values
    # we just compute the differences twice and find the point where the second difference is maximized
    inertias_diff = np.diff(inertias)
    inertias_ddiff = np.diff(inertias_diff)
    elbow_point = np.argmax(inertias_ddiff) + 2  # +2 to adjust for the double diff and zero-indexing
    optimal_k_inertia = k_range[elbow_point]
    print(f"Optimal number of clusters (k) based on elbow method: {optimal_k_inertia}")
    kmeans = KMeans(n_clusters=optimal_k_inertia, random_state=RANDOM_SEED)
    cluster_labels = kmeans.fit_predict(X_scaled)
    reporter.plot__clusters(
        X=X_scaled,
        labels=cluster_labels,
        title="KMeans Clustering (Inertia Optimal k)",
        algo_name="KMeans",
    )
    reporter.plot__world_map_clusters(df, cluster_labels, title="World Map KMeans Clusters (Inertia Optimal k)")
    reporter.plot__3d(X_scaled, cluster_labels, algo_name="KMeans", title="KMeans Clustering")


def countries_hands_on_training_dbscan(df: pd.DataFrame):
    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=3)
    cluster_labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"DBSCAN found {n_clusters} clusters.")

    reporter.plot__clusters(
        X=X_scaled,
        labels=cluster_labels,
        title="DBSCAN Clustering",
        algo_name="DBSCAN",
    )
    reporter.plot__world_map_clusters(df, cluster_labels, title="World Map DBSCAN Clusters")
    reporter.plot__3d(X_scaled, cluster_labels, algo_name="DBSCAN", title="DBSCAN Clustering")

def countries_hands_on_training_agglomerative(df: pd.DataFrame):
    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    agglomerative = AgglomerativeClustering(n_clusters=4)
    cluster_labels = agglomerative.fit_predict(X_scaled)

    reporter.plot__clusters(X_scaled, cluster_labels, title="Agglomerative Clustering", algo_name="Agglomerative")
    reporter.plot__world_map_clusters(df, cluster_labels, title="World Map Agglomerative Clustering")
    reporter.plot__3d(X_scaled, cluster_labels, algo_name="Agglomerative", title="Agglomerative Clustering")

def countries_hands_on_training_knn(df: pd.DataFrame):
    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal eps using k-distance graph
    n_neighbors = 7
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    model = neigh.fit(X_scaled)
    distances, indices = model.kneighbors(X_scaled)
    distances = np.sort(distances, axis=0)
    reporter.plot__k_distance_graph(n_neighbors=n_neighbors, distances=distances, title="k-Distance Graph")
    # from the graph, we would choose an eps value where there's a noticeable bend
    # for this example, let's assume we chose eps=0.5 (for previous iteration eps=0.5 was fine)
    dbscan = DBSCAN(eps=0.5, min_samples=n_neighbors)
    cluster_labels = dbscan.fit_predict(X_scaled)   
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"DBSCAN (with k-NN determined eps) found {n_clusters} clusters.")
    reporter.plot__clusters(X_scaled, cluster_labels, title="DBSCAN Clustering (k-NN eps)", algo_name="DBSCAN")
    reporter.plot__world_map_clusters(df, cluster_labels, title="World Map DBSCAN Clusters (k-NN eps)")

def countries_hands_on_training_optics_final(df: pd.DataFrame):
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    optics = OPTICS(eps=0.5, min_samples=7, xi=0.05, min_cluster_size=0.1)
    cluster_labels = optics.fit_predict(X_scaled)
    # just as DBSCAN, -1 indicates noise points (does not belong to any cluster) so we exclude it from the count
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"OPTICS found {n_clusters} clusters.")

    reporter.plot__clusters(X_scaled, cluster_labels, title="OPTICS Clustering", algo_name="OPTICS")
    reporter.plot__world_map_clusters(df, cluster_labels, title="World Map OPTICS Clusters")
    reporter.plot__3d(X_scaled, cluster_labels, algo_name="OPTICS", title="OPTICS Clustering")

def countries_hands_on_training_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Country Data Clustering Report", layout="wide")

    df = pd.read_csv(DATA_DIR / 'cleaned_Country_data.csv', index_col=0)
    df.reset_index(drop=True, inplace=True)
    df.set_index('country', inplace=True)
    compressed_df = pd.read_csv(DATA_DIR / 'compressed_country_data.csv', index_col=0)
    
    st.title("Country Data Clustering Report")
    st.header("Introduction")
    st.write("""
        This report presents an analysis of country data using various clustering algorithms. 
        The dataset includes multiple features related to countries' economic, health, and social indicators.
        We will explore the dataset, visualize distributions, and apply clustering techniques such as KMeans, DBSCAN, OPTICS, and Agglomerative Clustering.
    """)

    st.header("Data Exploration")
    st.subheader("Dataset Overview")
    st.dataframe(data=df)

    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    st.subheader("Numerical Features Distribution")
    for col in df.select_dtypes(include=np.number).columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=15, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.header("Dimensionality Reduction and Clustering")
    st.subheader("Compressed Data Overview")
    X = compressed_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.dataframe(data=compressed_df.values)

    st.subheader("Compressed Data Distribution")
    for col in compressed_df.columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(compressed_df[col], bins=15, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Clustering Visualizations -- KMeans Example")
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    st.plotly_chart(reporter.plot__3d(X_scaled, labels, algo_name="KMeans", title="KMeans Clustering"))

    st.subheader("Clustering Visualizations -- DBSCAN Example")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels = dbscan.fit_predict(X_scaled)
    st.plotly_chart(reporter.plot__3d(X_scaled, labels, algo_name="DBSCAN", title="DBSCAN Clustering"))

    st.subheader("Clustering Visualizations -- OPTICS Example")
    optics = OPTICS(eps=0.5, min_samples=7, xi=0.05, min_cluster_size=0.1)
    labels = optics.fit_predict(X_scaled)
    st.plotly_chart(reporter.plot__3d(X_scaled, labels, algo_name="OPTICS", title="OPTICS Clustering"))

    st.subheader("Clustering Visualizations -- Agglomerative clustering Example")
    agglomerative = AgglomerativeClustering(n_clusters=4)
    labels = agglomerative.fit_predict(X_scaled)
    st.plotly_chart(reporter.plot__3d(X_scaled, labels, algo_name="Agglomerative", title="Agglomerative Clustering"))

    st.subheader("Clustering Visualizations -- k-NN based DBSCAN Example")
    neigh = NearestNeighbors(n_neighbors=7)
    model = neigh.fit(X_scaled)
    distances, indices = model.kneighbors(X_scaled)
    distances = np.sort(distances, axis=0)
    fig, ax = plt.subplots()
    ax.plot(distances[:, 6])
    ax.set_title("k-NN Distance Graph (k=7)")
    st.pyplot(fig)
    dbscan_knn = DBSCAN(eps=0.5, min_samples=7)
    labels = dbscan_knn.fit_predict(X_scaled)
    st.plotly_chart(reporter.plot__3d(X_scaled, labels, algo_name="DBSCAN", title="k-NN based DBSCAN Clustering"))

    st.header("World Map Visualizations")
    st.subheader("KMeans Clusters on World Map")
    st.plotly_chart(reporter.plot__world_map_clusters(compressed_df, kmeans.labels_, title="World Map KMeans Clusters"))
    st.subheader("DBSCAN Clusters on World Map")
    st.plotly_chart(reporter.plot__world_map_clusters(compressed_df, dbscan.labels_, title="World Map DBSCAN Clusters"))
    st.subheader("OPTICS Clusters on World Map")
    st.plotly_chart(reporter.plot__world_map_clusters(compressed_df, optics.labels_, title="World Map OPTICS Clusters"))
    st.subheader("Agglomerative Clusters on World Map")
    st.plotly_chart(reporter.plot__world_map_clusters(compressed_df, agglomerative.labels_, title="World Map Agglomerative Clusters"))
    st.subheader("k-NN based DBSCAN Clusters on World Map")
    st.plotly_chart(reporter.plot__world_map_clusters(compressed_df, dbscan_knn.labels_, title="World Map k-NN based DBSCAN Clusters"))

    st.header("Conclusion")
    st.write("""
        In this report, we explored various clustering algorithms applied to country data. 
        Each algorithm provided unique insights into the data structure, highlighting different aspects of country similarities and differences.
    """)


def main():
    countries_hands_on_analysis()

    df = pd.read_csv(DATA_DIR / 'compressed_country_data.csv', index_col=0)
    
    countries_hands_on_training_kmeans(df)
    countries_hands_on_training_dbscan(df)
    countries_hands_on_training_agglomerative(df)
    countries_hands_on_training_knn(df)
    countries_hands_on_training_optics_final(df)

    countries_hands_on_training_streamlit()


if __name__ == "__main__":
    main()
