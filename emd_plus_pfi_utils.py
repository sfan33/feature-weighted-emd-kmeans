from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_pfi_weights(df, label_col, random_state=42, n_repeats=10):
    """
    Computes feature weights using Permutation Feature Importance (PFI).
    """
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()

    # Train a random forest classifier
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X, y)

    # Compute PFI
    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=random_state)
    importances = result.importances_mean

    # Normalize to sum to 1
    weights = importances / np.sum(importances)

    return weights

def compute_emd_plus_from_saved_perms_with_weights(df, perms_path, weights):
    data = df.to_numpy()
    perms_df = pd.read_csv(perms_path, index_col=0).astype(str).fillna("")

    row_size = data.shape[0]
    pairwise_emd_plus = np.zeros((row_size, row_size), dtype=float)

    with tqdm(total=row_size * (row_size - 1) // 2, desc="Computing EMD+ from saved permutations + PFI") as pbar:
        for i in range(row_size):
            row_i = data[i]
            for j in range(i + 1, row_size):
                row_j = data[j]
                perm_str = perms_df.iloc[i, j]
                if perm_str == "":
                    continue

                perm = np.array(list(map(int, perm_str.split(",")))) - 1  # fix 1-based to 0-based
                r_i = row_i[perm]
                r_j = row_j[perm]

                cdf_i = np.cumsum(r_i)
                cdf_j = np.cumsum(r_j)

                weighted_diff = np.abs(cdf_i - cdf_j) * weights[perm]
                emd_val = np.sum(weighted_diff)

                pairwise_emd_plus[i, j] = emd_val
                pairwise_emd_plus[j, i] = emd_val
                pbar.update(1)

    return pd.DataFrame(pairwise_emd_plus, index=range(1, row_size+1), columns=range(1, row_size+1))
