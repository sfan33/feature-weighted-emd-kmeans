from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from tqdm import tqdm

def best_emd_for_pair_weighted(row_i, row_j, perms, weights):
    row_i_extended = row_i[perms] * weights[perms]
    row_j_extended = row_j[perms] * weights[perms]

    cdf_i = np.cumsum(row_i_extended, axis=1)
    cdf_j = np.cumsum(row_j_extended, axis=1)

    all_emd = np.sum(np.abs(cdf_i - cdf_j), axis=1)
    best_idx = np.argmin(all_emd)

    return all_emd[best_idx], perms[best_idx]

def get_weighted_pca_weights(df, n_components=None):
    if n_components is None:
        n_components = df.shape[1]

    pca = PCA(n_components=n_components)
    pca.fit(df)

    variances = pca.explained_variance_ratio_         # shape: (n_components,)
    abs_components = np.abs(pca.components_)          # shape: (n_components, n_features)

    # Weighted average of absolute component loadings
    weighted = np.average(abs_components, axis=0, weights=variances)
    weights = weighted / weighted.sum()               # Normalize to sum to 1

    return weights

def compute_emd_plus_from_saved_perms(df, perms_path, n_components=None):
    data = df.to_numpy()
    weights = get_weighted_pca_weights(df, n_components=n_components)

    perms_df = pd.read_csv(perms_path, index_col=0).astype(str)
    perms_df = perms_df.fillna("")  # handle diagonal or missing

    row_size = data.shape[0]
    pairwise_emd_plus = np.zeros((row_size, row_size), dtype=float)

    with tqdm(total=row_size * (row_size - 1) // 2, desc="Computing EMD+ from saved permutations") as pbar:
        for i in range(row_size):
            row_i = data[i]
            for j in range(i + 1, row_size):
                row_j = data[j]
                perm_str = perms_df.iloc[i, j]

                if perm_str == "":
                    continue  # Skip diagonal or missing

                perm = np.array(list(map(int, perm_str.split(","))))

                # Permute rows
                r_i = row_i[perm]
                r_j = row_j[perm]

                # CDFs
                cdf_i = np.cumsum(r_i)
                cdf_j = np.cumsum(r_j)

                # Weighted difference
                weighted_diff = np.abs(cdf_i - cdf_j) * weights[perm]
                emd_val = np.sum(weighted_diff)

                pairwise_emd_plus[i, j] = emd_val
                pairwise_emd_plus[j, i] = emd_val
                pbar.update(1)

    emd_df = pd.DataFrame(pairwise_emd_plus, index=range(1, row_size+1), columns=range(1, row_size+1))
    return emd_df
