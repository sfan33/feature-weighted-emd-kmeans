import numpy as np
import pandas as pd
from tqdm import tqdm

def best_emd_for_pair(row_i, row_j, perms):
    # Permute row_i and row_j across all permutations (vectorized)
    row_i_extended = row_i[perms]
    row_j_extended = row_j[perms]

    # Compute the cumulative sums
    cdf_i = np.cumsum(row_i_extended, axis=1) 
    cdf_j = np.cumsum(row_j_extended, axis=1)
    all_emd = np.sum(np.abs(cdf_i - cdf_j), axis=1)
    best_idx = np.argmin(all_emd)

    return all_emd[best_idx], perms[best_idx]

def compute_pairwise_best_emd_and_perm(df, num_perms=None, random_state=None):
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random

    row_size, column_size = df.shape
    data = df.to_numpy()
    perms = np.array([rng.permutation(column_size) for _ in range(num_perms)]) # shape: (num_perms, m)
    pairwise_emd = np.zeros((row_size, row_size), dtype=float)
    pairwise_perms = np.empty((row_size, row_size), dtype=object)  # store best perms as Python objects/strings

    total_pairs = row_size * (row_size - 1) // 2
    with tqdm(total=total_pairs, desc="Computing pairwise best EMD") as pbar:
        for i in range(row_size):
            row_i = data[i]
            for j in range(i + 1, row_size):
                row_j = data[j]

                # Vectorized computation for all permutations
                best_emd_val, best_perm_arr = best_emd_for_pair(row_i, row_j, perms)
                pairwise_emd[i, j] = best_emd_val
                pairwise_emd[j, i] = best_emd_val
                
                best_perm_str = ",".join(map(str, best_perm_arr))
                pairwise_perms[i, j] = best_perm_str
                pairwise_perms[j, i] = best_perm_str
                pbar.update(1)

    emd_df = pd.DataFrame(pairwise_emd, index=range(1, row_size+1), columns=range(1, row_size+1))
    perms_df = pd.DataFrame(pairwise_perms, index=range(1, row_size+1), columns=range(1, row_size+1))
    
    return emd_df, perms_df
