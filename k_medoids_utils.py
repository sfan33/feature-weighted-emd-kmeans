import numpy as np
from scipy.spatial.distance import cdist

def kmedoids_euclidean(X, n_clusters=3, max_iter=100, random_state=42):
    distance_matrix = cdist(X, X, metric='euclidean')
    labels, medoids, total_cost = kmedoids_emd(
        distance_matrix,
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state
    )
    return labels, medoids, total_cost

def kmedoids_emd(distance_matrix, n_clusters, max_iter=100, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = distance_matrix.shape[0]
    medoids = rng.choice(n_samples, size=n_clusters, replace=False)

    # Store the cluster assignment for each point
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        for i in range(n_samples):
            distances_to_medoids = distance_matrix[i, medoids] # 1D array stores the distances from point i to the current medoids
            labels[i] = np.argmin(distances_to_medoids)

        updated = False
        for cluster_idx in range(n_clusters):
            cluster_points = np.where(labels == cluster_idx)[0]
            current_medoid = medoids[cluster_idx]
            current_cost = distance_matrix[cluster_points][:, current_medoid].sum()
            
            for candidate in cluster_points:
                if candidate == current_medoid:
                    continue

                candidate_cost = distance_matrix[cluster_points][:, candidate].sum()
                if candidate_cost < current_cost:
                    medoids[cluster_idx] = candidate
                    current_cost = candidate_cost
                    updated = True
        
        if not updated:
            break

    # Final assignment with the last medoids
    for i in range(n_samples):
        distances_to_medoids = distance_matrix[i, medoids]
        labels[i] = np.argmin(distances_to_medoids)

    total_cost = 0.0
    for i in range(n_samples):
        total_cost += distance_matrix[i, medoids[labels[i]]]

    return labels, medoids, total_cost
