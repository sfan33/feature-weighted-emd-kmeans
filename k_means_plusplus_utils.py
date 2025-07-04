import numpy as np
from sklearn.cluster import KMeans

def kmeans_plusplus_euclidean(X, n_clusters, random_state=None):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_

def kmeans_plusplus_emd_init(emd_matrix, n_clusters, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = emd_matrix.shape[0]
    centers = []
    centers.append(np.random.randint(n_samples))

    for _ in range(1, n_clusters):
        # Distance to the nearest center for each point
        min_dists = np.min(emd_matrix[:, centers], axis=1)
        probs = min_dists ** 2
        probs /= probs.sum()
        next_center = np.random.choice(n_samples, p=probs)
        centers.append(next_center)

    return np.array(centers)

def kmeans_plusplus_emd(emd_matrix, n_clusters, max_iter=100, random_state=None):
    centers_idx = kmeans_plusplus_emd_init(emd_matrix, n_clusters, random_state)
    labels = np.zeros(len(emd_matrix), dtype=int)

    for _ in range(max_iter):
        # Assign labels based on closest center
        for i in range(len(emd_matrix)):
            labels[i] = np.argmin([emd_matrix[i][c] for c in centers_idx])

        # Recompute centers as medoids (min total distance)
        new_centers = []
        for k in range(n_clusters):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                continue
            total_dist = emd_matrix[np.ix_(cluster_indices, cluster_indices)].sum(axis=1)
            new_center = cluster_indices[np.argmin(total_dist)]
            new_centers.append(new_center)

        if np.array_equal(centers_idx, new_centers):
            break
        centers_idx = new_centers

    return labels, centers_idx
