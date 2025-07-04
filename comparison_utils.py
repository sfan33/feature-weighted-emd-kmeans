from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score, 
    normalized_mutual_info_score, 
    adjusted_mutual_info_score,
    mutual_info_score,
    fowlkes_mallows_score, 
    homogeneity_score, 
    completeness_score, 
    v_measure_score
)
import pandas as pd
import numpy as np

def silhouette_score_from_distance_matrix(D, labels):
    n = len(labels)
    unique_labels = np.unique(labels)
    silhouette_vals = []

    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = unique_labels[unique_labels != labels[i]]

        # a: average distance to same cluster
        a = np.mean(D[i, same_cluster][D[i, same_cluster] > 0]) if np.sum(same_cluster) > 1 else 0

        # b: minimum average distance to other clusters
        b = np.inf
        for c in other_clusters:
            mask = labels == c
            dist = np.mean(D[i, mask]) if np.any(mask) else np.inf
            if dist < b:
                b = dist

        # silhouette value for point i
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_vals.append(s)

    return np.mean(silhouette_vals)

def davies_bouldin_from_distance_matrix(D, labels):
    clusters = np.unique(labels)
    k = len(clusters)

    centroids = []
    intra_dists = []
    
    for c in clusters:
        idx = np.where(labels == c)[0]
        intra = D[np.ix_(idx, idx)]
        # Use medoid (min sum of distances) as pseudo-centroid
        medoid_idx = idx[np.argmin(np.sum(intra, axis=1))]
        centroids.append(medoid_idx)

        # Average distance from medoid to other members
        avg_dist = np.mean(D[medoid_idx, idx])
        intra_dists.append(avg_dist)

    db_vals = []
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i == j:
                continue
            # Inter-cluster distance: between medoids
            inter_dist = D[centroids[i], centroids[j]]
            if inter_dist == 0:
                continue
            ratio = (intra_dists[i] + intra_dists[j]) / inter_dist
            max_ratio = max(max_ratio, ratio)
        db_vals.append(max_ratio)

    return np.mean(db_vals)

def calinski_harabasz_from_distance_matrix(D, labels):
    n = len(labels)
    clusters = np.unique(labels)
    k = len(clusters)

    total_mean_idx = np.argmin(np.sum(D, axis=1))  # global medoid
    intra_sum = 0
    inter_sum = 0

    for c in clusters:
        idx = np.where(labels == c)[0]
        intra = np.mean(D[np.ix_(idx, idx)])
        intra_sum += intra * len(idx)

        cluster_medoid = idx[np.argmin(np.sum(D[np.ix_(idx, idx)], axis=1))]
        inter = D[cluster_medoid, total_mean_idx]
        inter_sum += inter * len(idx)

    # Approximate CH score
    score = (inter_sum / (k - 1)) / (intra_sum / (n - k)) if (k - 1) > 0 and (n - k) > 0 else 0
    return score

def print_clustering_metrics_all_methods(X, true_labels, all_labels, save_path=None, emd_matrix=None):
    rows = []

    for method_name, labels in all_labels.items():
        print(f"\n=== {method_name} ===")
        row = {"Method": method_name}

        if "EMD" in method_name and emd_matrix is not None:
            try:
                row["Silhouette"] = silhouette_score_from_distance_matrix(emd_matrix, labels)
                print(f"Silhouette (EMD): {row['Silhouette']:.4f}")
            except Exception:
                row["Silhouette"] = None

            try:
                row["Davies-Bouldin"] = davies_bouldin_from_distance_matrix(emd_matrix, labels)
                print(f"Davies-Bouldin (EMD): {row['Davies-Bouldin']:.4f}")
            except Exception:
                row["Davies-Bouldin"] = None

            try:
                row["Calinski-Harabasz"] = calinski_harabasz_from_distance_matrix(emd_matrix, labels)
                print(f"Calinski-Harabasz (EMD): {row['Calinski-Harabasz']:.4f}")
            except Exception:
                row["Calinski-Harabasz"] = None
        else:
            try:
                row["Silhouette"] = silhouette_score(X, labels)
                print(f"Silhouette Score: {row['Silhouette']:.4f}")
            except ValueError:
                row["Silhouette"] = None
                
            try:
                row["Davies-Bouldin"] = davies_bouldin_score(X, labels)
                print(f"Davies-Bouldin Index: {row['Davies-Bouldin']:.4f}")
            except ValueError:
                row["Davies-Bouldin"] = None

            try:
                row["Calinski-Harabasz"] = calinski_harabasz_score(X, labels)
                print(f"Calinski-Harabasz Index: {row['Calinski-Harabasz']:.4f}")
            except ValueError:
                row["Calinski-Harabasz"] = None

        # External metrics
        if true_labels is not None:
            row["ARI"] = adjusted_rand_score(true_labels, labels)
            row["NMI"] = normalized_mutual_info_score(true_labels, labels)
            row["AMI"] = adjusted_mutual_info_score(true_labels, labels)
            row["MI"] = mutual_info_score(true_labels, labels)
            row["FMI"] = fowlkes_mallows_score(true_labels, labels)
            row["Homogeneity"] = homogeneity_score(true_labels, labels)
            row["Completeness"] = completeness_score(true_labels, labels)
            row["V-Measure"] = v_measure_score(true_labels, labels)

            print(f"ARI: {row['ARI']:.4f}")
            print(f"NMI: {row['NMI']:.4f}")
            print(f"AMI: {row['AMI']:.4f}")
            print(f"MI: {row['MI']:.4f}")
            print(f"FMI: {row['FMI']:.4f}")
            print(f"Homogeneity: {row['Homogeneity']:.4f}")
            print(f"Completeness: {row['Completeness']:.4f}")
            print(f"V-Measure: {row['V-Measure']:.4f}")
        else:
            print("Ground truth not provided. Skipping external metrics.")
            for key in ["ARI", "NMI", "AMI", "MI", "FMI", "Homogeneity", "Completeness", "V-Measure"]:
                row[key] = None

        print("-" * 40)
        rows.append(row)

    df_metrics = pd.DataFrame(rows)
    
    if save_path:
        df_metrics.to_csv(save_path, index=False)
        print(f"Clustering metrics saved to '{save_path}'.")

    return df_metrics
