import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def visualize_tsne(df, title="t-SNE 2D", random_state=42, perplexity=30.0, save_path=None):
    """
    Applies t-SNE to 'df' (rows are samples, columns are features).
    - n_components=2: embed data into 2D
    - perplexity: hyperparameter controlling local/global weighting
    - random_state for reproducibility
    - save_path: if provided, the plot is saved to this path (no interactive popup).
    """
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        random_state=random_state
    )
    coords_2d = tsne.fit_transform(df.values)

    plt.figure()
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # close the figure so it doesn't display
    else:
        plt.show()

def plot_distance_heatmap(df, title="Distance Heatmap", save_path=None):
    """
    Computes a simple pairwise Euclidean distance matrix among rows of 'df'
    and visualizes it as a heatmap.
    - save_path: if provided, save to this path (no interactive popup).
    """
    n = len(df)
    dist_matrix = np.zeros((n, n))
    data_arr = df.values

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(data_arr[i] - data_arr[j])

    plt.figure()
    plt.imshow(dist_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label="Distance")
    plt.title(title)
    plt.xlabel("Row index")
    plt.ylabel("Row index")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def cluster_and_parallel_coords(df, n_clusters=3, title="Parallel Coordinates", save_path=None):
    """
    Runs KMeans clustering on 'df' (rows x features) to get cluster labels.
    Then creates a Parallel Coordinates plot to visualize all columns at once,
    with one vertical axis per feature.
    
    Arguments:
    - df: pandas DataFrame with numeric features (rows = samples)
    - n_clusters: how many clusters for KMeans
    - title: title for the plot
    - save_path: optional path to save the figure to disk (png, etc.).
      If None, the plot is shown interactively.
    """
    # 1) Fit KMeans to get cluster labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df)

    # 2) Combine the original data with the cluster labels
    df_plot = df.copy()
    df_plot["cluster"] = labels  # so we can color lines by cluster

    # 3) Create a parallel coordinates plot
    plt.figure(figsize=(10, 6))  # adjust size to fit all features
    parallel_coordinates(df_plot, class_column="cluster", colormap='viridis')

    plt.title(title)
    plt.xticks(rotation=45)  # rotate feature names on X-axis if needed
    plt.tight_layout()

    # 4) Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        