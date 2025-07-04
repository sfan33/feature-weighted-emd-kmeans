from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_metric(metric_list, higher_is_better=True):
    arr = np.array(metric_list, dtype=np.float64)
    mask = ~np.isnan(arr)
    if higher_is_better:
        normed = MinMaxScaler().fit_transform(arr[mask].reshape(-1, 1)).flatten()
    else:
        normed = MinMaxScaler().fit_transform(-arr[mask].reshape(-1, 1)).flatten()
    result = np.full_like(arr, fill_value=np.nan)
    result[mask] = normed
    return result

def evaluate_kmeans_and_select_best(df, k_range=range(2, 11), save_plot_path=None, save_metrics_path=None):
    sse, silhouette_scores, ch_scores, db_scores, sil_stddevs = [], [], [], [], []
    kmeans_models = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(df)
        kmeans_models.append((k, kmeans))

        # SSE
        sse.append(kmeans.inertia_)

        try:
            # Silhouette Score
            sil_values = silhouette_score(df, labels)
            silhouette_scores.append(sil_values)

            # Silhouette StdDev (simulated; to get true std dev use silhouette_samples)
            sil_stddevs.append(np.std([sil_values]))  # You can replace this later with silhouette_samples
        except Exception as e:
            print(f"Silhouette score failed at k={k}: {e}")
            silhouette_scores.append(np.nan)
            sil_stddevs.append(np.nan)

        try:
            ch_scores.append(calinski_harabasz_score(df, labels))
        except Exception as e:
            print(f"CH score failed at k={k}: {e}")
            ch_scores.append(np.nan)

        try:
            db_scores.append(davies_bouldin_score(df, labels))
        except Exception as e:
            print(f"DB index failed at k={k}: {e}")
            db_scores.append(np.nan)

    results = pd.DataFrame({
        "k": list(k_range),
        "SSE": sse,
        "Silhouette": silhouette_scores,
        "Silhouette StdDev": sil_stddevs,
        "Calinski-Harabasz": ch_scores,
        "Davies-Bouldin": db_scores,
    })

    # Normalize metrics
    sse_norm = normalize_metric(sse, higher_is_better=False)
    silhouette_norm = normalize_metric(silhouette_scores, higher_is_better=True)
    sil_std_norm = normalize_metric(sil_stddevs, higher_is_better=False)
    ch_norm = normalize_metric(ch_scores, higher_is_better=True)
    db_norm = normalize_metric(db_scores, higher_is_better=False)

    # Combine score (average of all 5 normalized scores)
    combined_score = (sse_norm + silhouette_norm + ch_norm + db_norm + sil_std_norm) / 5
    results["CombinedScore"] = combined_score

    # Find best k
    best_k_index = np.nanargmax(combined_score)
    best_k = k_range[best_k_index]
    best_model = kmeans_models[best_k_index][1]

    # Plotting
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title("SSE")
    plt.xlabel("k")

    plt.subplot(1, 5, 2)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("k")

    plt.subplot(1, 5, 3)
    plt.plot(k_range, ch_scores, marker='o')
    plt.title("Calinski-Harabasz")
    plt.xlabel("k")

    plt.subplot(1, 5, 4)
    plt.plot(k_range, db_scores, marker='o')
    plt.title("Davies-Bouldin")
    plt.xlabel("k")

    plt.subplot(1, 5, 5)
    plt.plot(k_range, sil_stddevs, marker='o')
    plt.title("Silhouette StdDev")
    plt.xlabel("k")

    plt.tight_layout()
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved k-means evaluation plot to '{save_plot_path}'")
    plt.close()

    if save_metrics_path:
        results.to_csv(save_metrics_path, index=False)
        print(f"Saved k-means evaluation metrics to '{save_metrics_path}'")

    return best_k, best_model, results

def visualize_clusters_tsne(df, labels, title="t-SNE Clustering", save_path="tsne_clusters.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cluster'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10')
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE cluster plot to '{save_path}'")

    return tsne_df
