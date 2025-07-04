from sklearn.cluster import MiniBatchKMeans
import numpy as np

def minibatch_kmeans_euclidean(X, n_clusters=3, batch_size=32, max_iter=100, random_state=None):
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state
    )
    mbk.fit(X)
    return mbk.labels_, mbk.cluster_centers_, mbk.inertia_

def minibatch_kmeans_emd(emd_matrix, n_clusters=3, batch_size=10, max_iter=50, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = emd_matrix.shape[0]
    centroids = [np.random.randint(n_samples)]
    # K-Means++ style init for the rest
    while len(centroids) < n_clusters:
        min_dists = np.min(emd_matrix[:, centroids], axis=1)
        probs = min_dists ** 2
        probs /= probs.sum()
        next_centroid = np.random.choice(n_samples, p=probs)
        centroids.append(next_centroid)

    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iter):
        # Mini-batch sampling
        minibatch = np.random.choice(n_samples, size=batch_size, replace=False)
        for i in minibatch:
            dists = [emd_matrix[i, c] for c in centroids]
            labels[i] = np.argmin(dists)

        # Update step: for each cluster, find new centroid (medoid-style)
        new_centroids = []
        for k in range(n_clusters):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                new_centroids.append(np.random.randint(n_samples))
                continue
            submatrix = emd_matrix[np.ix_(cluster_indices, cluster_indices)]
            total_dist = submatrix.sum(axis=1)
            medoid_idx = cluster_indices[np.argmin(total_dist)]
            new_centroids.append(medoid_idx)

        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids
