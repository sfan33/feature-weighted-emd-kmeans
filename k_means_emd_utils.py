from emd_utils import best_emd_for_pair
import numpy as np
import ot

def emd_barycenter(distributions, cost_matrix=None, max_iter=5000, tol=1e-9, verbose=False):
    distributions = np.asarray(distributions, dtype=np.float64)
    distributions /= distributions.sum(axis=1, keepdims=True) # Normalizes each row to ensure all inputs are valid probability distributions

    A = distributions.T  # shape (dim, n_hists)
    assert not np.any(np.isnan(A)), "Distributions contain NaNs"
    assert not np.any(np.isnan(cost_matrix)), "Cost matrix contains NaNs"

    # Try increasingly stronger regularization values
    for reg_try in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 100.0, 200.0]:
        try:
            bary = ot.barycenter(
                A,
                cost_matrix,
                reg=reg_try,
                method='sinkhorn',
                numItermax=max_iter,
                stopThr=tol
            )
            bary = np.maximum(bary, 1e-12)  # avoid zeros
            bary /= bary.sum()

            if not np.any(np.isnan(bary)) and np.isfinite(bary).all():
                return bary
        except Exception as e:
            if verbose:
                print(f"[Error] Barycenter failed with reg={reg_try}: {e}")

    raise RuntimeError("Barycenter computation failed with all regularization settings.")

def kmeans_emd_with_permutations(distributions, n_clusters=3, num_perms=1000, max_iter=100, random_state=42, verbose=False):
    np.random.seed(random_state)
    n_samples, n_bins = distributions.shape
    rng = np.random.default_rng(random_state)
    perms = np.array([rng.permutation(n_bins) for _ in range(num_perms)])
    cost_matrix = np.abs(np.arange(n_bins).reshape(-1, 1) - np.arange(n_bins).reshape(1, -1)) # Cost matrix for EMD in 1D

    initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centers = [distributions[i].copy() for i in initial_indices]
    labels = np.zeros(n_samples, dtype=int)
    perms_used = [np.arange(n_bins) for _ in range(n_clusters)] # keep track of the best permutation found for each cluster

    for iteration in range(max_iter):
        changed = False
        for i in range(n_samples):
            row_i = distributions[i]
            dists = [best_emd_for_pair(row_i, centers[c], perms)[0] for c in range(n_clusters)]
            new_label = np.argmin(dists)
            if labels[i] != new_label:
                changed = True
                labels[i] = new_label

        for c in range(n_clusters):
            cluster_members = distributions[labels == c]
            if len(cluster_members) == 0:
                continue

            permuted_members = []
            for member in cluster_members:
                _, best_perm = best_emd_for_pair(member, centers[c], perms)
                permuted_members.append(member[best_perm])
                perms_used[c] = best_perm

            permuted_members = np.vstack(permuted_members) # Convert the list of permuted members into a matrix (rows = samples, cols = bins)
            new_center = emd_barycenter(permuted_members, cost_matrix=cost_matrix, verbose=verbose)
            centers[c] = new_center

        if not changed:
            if verbose:
                print(f"[KMeans-EMD] Converged after {iteration+1} iterations.")
            break

    return labels, centers, perms_used
