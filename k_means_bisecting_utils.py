from sklearn.cluster import KMeans
import numpy as np

def bisecting_kmeans_euclidean(X, n_clusters=3, max_iter=10, random_state=None):
    clusters = [np.arange(len(X))]
    labels = np.zeros(len(X), dtype=int)

    while len(clusters) < n_clusters:
        # Choose cluster with highest inertia
        inertias = []
        for idxs in clusters:
            if len(idxs) <= 1:
                inertias.append(0)
                continue
            km = KMeans(n_clusters=1).fit(X[idxs])
            inertias.append(km.inertia_)
        split_idx = np.argmax(inertias)
        to_split = clusters.pop(split_idx)

        # Bisect it using KMeans(k=2)
        if len(to_split) <= 1:
            clusters.append(to_split)
            continue

        km = KMeans(n_clusters=2, random_state=random_state, max_iter=max_iter).fit(X[to_split])
        labels[to_split] = len(clusters)  # temporary label for later reassignment
        left = to_split[km.labels_ == 0]
        right = to_split[km.labels_ == 1]
        clusters.append(left)
        clusters.append(right)

    # Reassign final labels
    final_labels = np.zeros(len(X), dtype=int)
    for i, idxs in enumerate(clusters):
        final_labels[idxs] = i

    return final_labels

def bisecting_kmeans_emd(emd_matrix, n_clusters=3, max_iter=10, random_state=None):
    n = len(emd_matrix)
    clusters = [np.arange(n)]
    _labels = np.zeros(n, dtype=int)

    def cluster_cost(indices):
        if len(indices) <= 1:
            return 0
        return emd_matrix[np.ix_(indices, indices)].sum()

    while len(clusters) < n_clusters:
        costs = [cluster_cost(c) for c in clusters]
        split_idx = np.argmax(costs)
        to_split = clusters.pop(split_idx)

        if len(to_split) <= 1:
            clusters.append(to_split)
            continue

        # Simple k-medoids with k=2 on the submatrix
        submatrix = emd_matrix[np.ix_(to_split, to_split)]
        m, n_local = submatrix.shape
        medoid1, medoid2 = np.random.choice(n_local, 2, replace=False)
        for _ in range(max_iter):
            labels_local = np.array([
                0 if submatrix[i, medoid1] < submatrix[i, medoid2] else 1
                for i in range(n_local)
            ])
            new_medoids = []
            for k in [0, 1]:
                cluster_i = np.where(labels_local == k)[0]
                if len(cluster_i) == 0:
                    continue
                total_dist = submatrix[np.ix_(cluster_i, cluster_i)].sum(axis=1)
                new_medoids.append(cluster_i[np.argmin(total_dist)])
            if len(new_medoids) < 2 or new_medoids == [medoid1, medoid2]:
                break
            medoid1, medoid2 = new_medoids

        left = to_split[labels_local == 0]
        right = to_split[labels_local == 1]
        clusters.append(left)
        clusters.append(right)

    # Reassign final labels
    final_labels = np.zeros(n, dtype=int)
    for i, idxs in enumerate(clusters):
        final_labels[idxs] = i

    return final_labels
