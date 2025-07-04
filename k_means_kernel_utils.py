import numpy as np

from sklearn.metrics.pairwise import pairwise_kernels

def compute_rbf_kernel(X, gamma=0.5):
    return pairwise_kernels(X, metric='rbf', gamma=gamma)

def compute_rbf_kernel_from_distances(D, gamma=0.5):
    return np.exp(-gamma * D ** 2)

def kernel_kmeans(K, n_clusters, max_iter=100, random_state=None):
    n_samples = K.shape[0]
    if random_state:
        np.random.seed(random_state)
    labels = np.random.randint(0, n_clusters, size=n_samples)

    for _ in range(max_iter):
        dist = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            cluster_idx = np.where(labels == j)[0]
            if len(cluster_idx) == 0:
                continue
            K_cluster = K[np.ix_(cluster_idx, cluster_idx)]
            denom = len(cluster_idx)
            term1 = np.diag(K)
            term2 = -2 * K[:, cluster_idx].sum(axis=1) / denom
            term3 = K_cluster.sum() / (denom ** 2)
            dist[:, j] = term1 + term2 + term3
        new_labels = np.argmin(dist, axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels
    return labels
