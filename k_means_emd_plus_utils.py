import numpy as np
from sklearn.utils import check_random_state

def _parse_perm_string(perm_str: str, n_features: int) -> np.ndarray:
    """
    Convert a permutation string from `perms_df` into a valid 0-based
    permutation of length `n_features`.  Any problem → identity permutation.
    """
    # Treat empty / null-like entries as identity
    if not perm_str or perm_str.lower() in {"none", "nan"}:
        return np.arange(n_features)

    try:
        # Fast parsing: "0,2,1,3,4"
        perm = np.fromstring(perm_str, sep=",", dtype=int)

        # Was it stored as 1-based?  (exact length and max == n_features)
        if (
            perm.size == n_features
            and perm.min() == 1
            and perm.max() == n_features
        ):
            perm -= 1

        # Validate: must be a permutation of 0 … n_features-1
        if (
            perm.size == n_features
            and np.unique(perm).size == n_features
            and perm.min() == 0
            and perm.max() == n_features - 1
        ):
            return perm.astype(int)
    except Exception:
        pass  # fall through to identity

    # Anything that reaches here is malformed → identity
    return np.arange(n_features)


def kmeans_emd_plus_with_permutations(
    X,
    n_clusters,
    perms_df,
    weights,
    max_iter=100,
    random_state=None,
    verbose=False,
):
    """
    K-Means-style clustering under EMD+ with pre-computed permutations.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Input distributions.
    n_clusters : int
        Desired number of clusters.
    perms_df : DataFrame
        n_samples × n_clusters table of permutation strings.
    weights : 1-D array-like length n_features
        Attribute weights for EMD+.
    max_iter : int, default 100
        Maximum number of label/centroid updates.
    random_state : int or RandomState, optional
        Reproducibility.
    verbose : bool, default False
        Print iteration progress.

    Returns
    -------
    labels : ndarray (n_samples,)
        Cluster assignment for each sample.
    centroids : ndarray (n_clusters, n_features)
        Final centroid distributions.
    """
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_samples, n_features = X.shape
    assert weights.shape == (n_features,), "weights must match n_features"

    # ---- initial centroids (random rows) ----
    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[init_idx].copy()

    # Permutation strings → numpy array of objects (fast row access)
    perms_arr = perms_df.astype(str).fillna("").values

    labels = np.full(n_samples, -1, dtype=int)

    for it in range(max_iter):
        # -------- assignment step --------
        new_labels = np.empty(n_samples, dtype=int)

        for i in range(n_samples):
            dists = np.empty(n_clusters)

            for k in range(n_clusters):
                perm = _parse_perm_string(perms_arr[i, k], n_features)

                # Apply permutation once to both vectors
                r_i = X[i, perm]
                r_c = centroids[k, perm]

                # Weighted 1-D EMD (L1 distance between CDFs)
                diff = np.abs(np.cumsum(r_i) - np.cumsum(r_c))
                dists[k] = np.dot(diff, weights[perm])

            new_labels[i] = int(np.argmin(dists))

        # Convergence check
        if np.array_equal(new_labels, labels):
            if verbose:
                print(f"Converged at iter {it}")
            break

        labels = new_labels

        # -------- update step --------
        for k in range(n_clusters):
            members = X[labels == k]
            if members.size:        # avoid empty slice warning
                centroids[k] = members.mean(axis=0)

    return labels, centroids