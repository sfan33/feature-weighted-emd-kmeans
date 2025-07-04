from pathlib import Path
from data_utils import load_dataset
from excel_utils import write_metrics_to_excel
from emd_plus_pfi_utils import (
    compute_emd_plus_from_saved_perms_with_weights,
    get_pfi_weights,
)
from feature_weight_utils import (
    get_lime_weights,
    get_rf_weights,
    get_shap_weights,
    get_xgb_weights,
)
from preprocess_utils import preprocess_for_pairwise
from compare import plot_distance_heatmap, cluster_and_parallel_coords
from comparison_utils import print_clustering_metrics_all_methods
from emd_utils import compute_pairwise_best_emd_and_perm
from emd_plus_pca_utils import (
    compute_emd_plus_from_saved_perms,
    get_weighted_pca_weights,
)
from k_means_utils import evaluate_kmeans_and_select_best
from k_means_emd_utils import kmeans_emd_with_permutations
from k_means_emd_plus_utils import kmeans_emd_plus_with_permutations
from k_medoids_utils import kmedoids_emd, kmedoids_euclidean
from k_means_plusplus_utils import kmeans_plusplus_euclidean, kmeans_plusplus_emd
from k_means_mini_batch_utils import minibatch_kmeans_euclidean, minibatch_kmeans_emd
from k_means_bisecting_utils import bisecting_kmeans_euclidean, bisecting_kmeans_emd
from k_means_kernel_utils import (
    kernel_kmeans,
    compute_rbf_kernel,
    compute_rbf_kernel_from_distances,
)


def main(file_path, sheet_name=None, value_col=None, id_col=None):
    df_original = load_dataset(file_path, sheet_name)
    df_original.dropna(inplace=True)

    if id_col and id_col in df_original.columns:
        print(f"Dropping identifier column '{id_col}' from the dataset.")
        df_original = df_original.drop(columns=[id_col])

    binary_value_col,df_target = preprocess_for_pairwise(
        df_original, encoding_method="target", label_col=value_col
    )
    plot_distance_heatmap(
        df_target,
        title="Target Distance Heatmap",
        save_path="results/dist_heatmap_target.png",
    )

    print("Finding best k and fitting standard KMeans...")
    best_k, best_model, clustering_metrics = evaluate_kmeans_and_select_best(
        df_target,
        save_plot_path="results/kmeans_evaluation.png",
        save_metrics_path="results/kmeans_metrics.csv",
    )
    print(f"Best k : {best_k}")
    cluster_and_parallel_coords(
        df_target,
        n_clusters=best_k,
        title="Parallel Coordinates - Target",
        save_path="results/parallel_coords_target.png",
    )

    labels_euclid = best_model.labels_

    print("\nComputing EMD and running K-Medoids-EMD...")
    num_perms = 100
    emd_df, perms_df = compute_pairwise_best_emd_and_perm(
        df_target, num_perms=num_perms, random_state=42
    )
    emd_df.to_csv("results/pairwise_emd.csv", index=True)
    print("Saved EMD matrix to 'results/pairwise_emd.csv'.")
    perms_df.to_csv("results/pairwise_best_perms.csv", index=True)
    print("Saved best permutations to 'results/pairwise_best_perms.csv'.\n")

    emd_matrix = emd_df.values
    X = df_target.values
    print(f"X shape: {X.shape}")
    labels_emd_medoids, medoids_emd, total_cost_emd = kmedoids_emd(
        distance_matrix=emd_matrix, n_clusters=best_k, max_iter=100, random_state=42
    )
    # visualize_clusters_tsne(df_target, labels_emd_medoids, save_path="results/tsne_kmedoids_emd_clusters.png", title="t-SNE Clusters (K-Medoids EMD)")
    print("K-Medoids-EMD completed.\n")

    print("\nComputing EMD+ from saved permutations...")
    emd_plus_pca_from_perms_df = compute_emd_plus_from_saved_perms(
        df_target, "results/pairwise_best_perms.csv", n_components=None
    )
    emd_plus_pca_from_perms_df.to_csv(
        "results/pairwise_emd_plus_from_saved.csv", index=True
    )
    print(
        "Saved EMD+ PCA (from saved permutations) to 'results/pairwise_emd_plus_from_saved.csv'.\n"
    )
    emd_plus_pca_matrix = emd_plus_pca_from_perms_df.values

    print("Computing PFI-based weights...")
    df_target_with_lbl = df_target.copy()
    df_target_with_lbl[value_col] = binary_value_col.values
    pfi_weights = get_pfi_weights(df_target_with_lbl, label_col=value_col, random_state=42)
    print("\nComputing EMD+ (PFI-weighted) from saved permutations...")
    emd_plus_pfi_df = compute_emd_plus_from_saved_perms_with_weights(
        df_target, "results/pairwise_best_perms.csv", pfi_weights
    )
    emd_plus_pfi_df.to_csv("results/pairwise_emd_plus_pfi.csv", index=True)
    print("Saved EMD+ (PFI-weighted) to 'results/pairwise_emd_plus_pfi.csv'.\n")
    emd_plus_pfi_matrix = emd_plus_pfi_df.values

    print("Computing RF-based weights...")
    rf_weights = get_rf_weights(df_target_with_lbl, label_col=value_col)
    print("\nComputing EMD+ (RF-weighted) from saved permutations...")
    emd_plus_rf_df = compute_emd_plus_from_saved_perms_with_weights(
        df_target, "results/pairwise_best_perms.csv", rf_weights
    )
    emd_plus_rf_df.to_csv("results/pairwise_emd_plus_rf.csv", index=True)
    print("Saved EMD+ (RF-weighted) to 'results/pairwise_emd_plus_rf.csv'.\n")
    emd_plus_rf_matrix = emd_plus_rf_df.values

    print("Computing XGBoost-based weights...")
    xgb_weights = get_xgb_weights(df_target_with_lbl, label_col=value_col)
    print("\nComputing EMD+ (XGBoost-weighted) from saved permutations...")
    emd_plus_xgb_df = compute_emd_plus_from_saved_perms_with_weights(
        df_target, "results/pairwise_best_perms.csv", xgb_weights
    )
    emd_plus_xgb_df.to_csv("results/pairwise_emd_plus_xgb.csv", index=True)
    print("Saved EMD+ (XGBoost-weighted) to 'results/pairwise_emd_plus_xgb.csv'.\n")
    emd_plus_xgb_matrix = emd_plus_xgb_df.values

    print("Computing SHAP-based weights...")
    shap_weights = get_shap_weights(
        df_target_with_lbl, label_col=value_col, random_state=42
    )
    print("SHAP weights shape:", shap_weights.shape, "sum:", shap_weights.sum())
    print("\nComputing EMD+ (SHAP-weighted) from saved permutations...")
    emd_plus_shap_df = compute_emd_plus_from_saved_perms_with_weights(
        df_target, "results/pairwise_best_perms.csv", shap_weights
    )
    emd_plus_shap_df.to_csv("results/pairwise_emd_plus_shap.csv", index=True)
    print("Saved EMD+ (SHAP-weighted) to 'results/pairwise_emd_plus_shap.csv'.\n")
    emd_plus_shap_matrix = emd_plus_shap_df.values

    print("Computing LIME-based weights...")
    lime_weights = get_lime_weights(df_target_with_lbl, label_col=value_col, num_samples=100)
    print("\nComputing EMD+ (LIME-weighted) from saved permutations...")
    emd_plus_lime_df = compute_emd_plus_from_saved_perms_with_weights(
        df_target, "results/pairwise_best_perms.csv", lime_weights
    )
    emd_plus_lime_df.to_csv("results/pairwise_emd_plus_lime.csv", index=True)
    print("Saved EMD+ (LIME-weighted) to 'results/pairwise_emd_plus_lime.csv'.\n")
    emd_plus_lime_matrix = emd_plus_lime_df.values

    distributions = df_target.values.astype(float)
    labels_emd_kmeans, centers_emd, perms_used = kmeans_emd_with_permutations(
        distributions,
        n_clusters=best_k,
        num_perms=100,
        max_iter=50,
        random_state=42,
        verbose=False,
    )
    print("K-Means-style EMD completed.\n")

    labels_emd_kmeans_plus_pca, centers_emd_plus_pca = (
        kmeans_emd_plus_with_permutations(
            X=distributions,
            n_clusters=best_k,
            perms_df=perms_df,
            weights=get_weighted_pca_weights(df_target),
            max_iter=50,
            random_state=42,
            verbose=False
        )
    )

    labels_emd_kmeans_plus_pfi, centers_emd_plus_pfi = (
        kmeans_emd_plus_with_permutations(
            X=distributions,
            n_clusters=best_k,
            perms_df=perms_df,
            weights=pfi_weights,
            max_iter=50,
            random_state=42,
            verbose=False,
        )
    )

    labels_emd_kmeans_plus_rf, centers_emd_plus_rf = (
        kmeans_emd_plus_with_permutations(
            X=distributions,
            n_clusters=best_k,
            perms_df=perms_df,
            weights=rf_weights,
            max_iter=50,
            random_state=42,
            verbose=False,
        )
    )

    labels_emd_kmeans_plus_xgb, centers_emd_plus_xgb = (
        kmeans_emd_plus_with_permutations(
            X=distributions,
            n_clusters=best_k,
            perms_df=perms_df,
            weights=xgb_weights,
            max_iter=50,
            random_state=42,
            verbose=False,
        )
    )
    labels_emd_kmeans_plus_shap, centers_emd_plus_shap = (
        kmeans_emd_plus_with_permutations(
            X=distributions,
            n_clusters=best_k,
            perms_df=perms_df,
            weights=shap_weights,
            max_iter=50,
            random_state=42,
            verbose=False,
        )
    )
    labels_emd_kmeans_plus_lime, centers_emd_plus_lime = (
        kmeans_emd_plus_with_permutations(
            X=distributions,
            n_clusters=best_k,
            perms_df=perms_df,
            weights=lime_weights,
            max_iter=50,
            random_state=42,
            verbose=False,
        )
    )
    print("K-Means-style EMD+ completed.\n")

    labels_emd_plus_pfi_medoids, medoids_emd_plus_pfi, _ = kmedoids_emd(
        distance_matrix=emd_plus_pfi_matrix,
        n_clusters=best_k,
        max_iter=100,
        random_state=42,
    )
    labels_emd_plus_pca_medoids, medoids_emd_plus, cost_emd_plus = kmedoids_emd(distance_matrix=emd_plus_pca_matrix,
        n_clusters=best_k,
        max_iter=100,
        random_state=42,
    )
    labels_emd_plus_rf_medoids, _, _ = kmedoids_emd(
        emd_plus_rf_matrix, best_k, max_iter=100, random_state=42
    )
    labels_emd_plus_xgb_medoids, _, _ = kmedoids_emd(
        emd_plus_xgb_matrix, best_k, max_iter=100, random_state=42
    )
    labels_emd_plus_shap_medoids, _, _ = kmedoids_emd(
        emd_plus_shap_matrix, best_k, max_iter=100, random_state=42
    )
    labels_emd_plus_lime_medoids, _, _ = kmedoids_emd(
        emd_plus_lime_matrix, best_k, max_iter=100, random_state=42
    )
    print("K-Medoids with EMD+ (saved perms) completed.\n")

    labels_euclid_medoids, medoids_euclid, cost_euclid = kmedoids_euclidean(
        X=X, n_clusters=best_k, max_iter=100, random_state=42
    )
    print("K-Medoids (Euclidean) completed.\n")

    labels_kpp_euclid, centers_kpp, inertia_kpp = kmeans_plusplus_euclidean(
        X, best_k, random_state=42
    )
    print("K-Means++ with Euclidean completed.\n")

    labels_kpp_emd, centers_idx_kpp_emd = kmeans_plusplus_emd(
        emd_matrix, best_k, random_state=42
    )
    print("K-Means++ with EMD completed.\n")

    labels_kpp_emd_plus_pca, centers_idx_kpp_emd_plus = kmeans_plusplus_emd(
        emd_plus_pca_matrix, best_k, random_state=42
    )
    labels_kpp_emd_plus_pfi, centers_idx_kpp_emd_plus = kmeans_plusplus_emd(
        emd_plus_pfi_matrix, best_k, random_state=42
    )
    labels_kpp_emd_plus_rf, centers_idx_kpp_emd_plus = kmeans_plusplus_emd(
        emd_plus_rf_matrix, best_k, random_state=42
    )
    labels_kpp_emd_plus_xgb, centers_idx_kpp_emd_plus = kmeans_plusplus_emd(
        emd_plus_xgb_matrix, best_k, random_state=42
    )
    labels_kpp_emd_plus_shap, centers_idx_kpp_emd_plus = kmeans_plusplus_emd(
        emd_plus_shap_matrix, best_k, random_state=42
    )
    labels_kpp_emd_plus_lime, centers_idx_kpp_emd_plus = kmeans_plusplus_emd(
        emd_plus_lime_matrix, best_k, random_state=42
    )
    print("K-Means++ with EMD+ completed.\n")

    labels_mbk_euclid, centers_mbk, inertia_mbk = minibatch_kmeans_euclidean(
        X, n_clusters=best_k, batch_size=32, max_iter=100, random_state=42
    )
    print("Mini-Batch K-Means with Euclidean completed.\n")

    labels_mbk_emd, centers_mbk_emd = minibatch_kmeans_emd(
        emd_matrix, n_clusters=best_k, batch_size=10, max_iter=100, random_state=42
    )
    print("Mini-Batch K-Means with EMD completed.\n")

    labels_mbk_emd_plus_pca, centers_mbk_emd_plus = minibatch_kmeans_emd(
        emd_plus_pca_matrix,
        n_clusters=best_k,
        batch_size=10,
        max_iter=100,
        random_state=42,
    )
    labels_mbk_emd_plus_pfi, centers_mbk_emd_plus = minibatch_kmeans_emd(
        emd_plus_pfi_matrix,
        n_clusters=best_k,
        batch_size=10,
        max_iter=100,
        random_state=42,
    )
    labels_mbk_emd_plus_rf, centers_mbk_emd_plus = minibatch_kmeans_emd(
        emd_plus_rf_matrix,
        n_clusters=best_k,
        batch_size=10,
        max_iter=100,
        random_state=42,
    )
    labels_mbk_emd_plus_xgb, centers_mbk_emd_plus = minibatch_kmeans_emd(
        emd_plus_xgb_matrix,
        n_clusters=best_k,
        batch_size=10,
        max_iter=100,
        random_state=42,)
    labels_mbk_emd_plus_shap, centers_mbk_emd_plus = minibatch_kmeans_emd(emd_plus_shap_matrix, n_clusters=best_k, batch_size=10, max_iter=100, random_state=42)
    labels_mbk_emd_plus_lime, centers_mbk_emd_plus = minibatch_kmeans_emd(emd_plus_lime_matrix, n_clusters=best_k, batch_size=10, max_iter=100, random_state=42,)
    print("Mini-Batch K-Means with EMD+ completed.\n")

    labels_bisect_euclid = bisecting_kmeans_euclidean(
        X, n_clusters=best_k, random_state=42
    )
    print("Bisecting K-Means with Euclidean completed.\n")

    labels_bisect_emd = bisecting_kmeans_emd(
        emd_matrix, n_clusters=best_k, random_state=42
    )
    print("Bisecting K-Means with EMD completed.\n")

    labels_bisect_emd_plus_pca = bisecting_kmeans_emd(
        emd_plus_pca_matrix, n_clusters=best_k, random_state=42
    )
    labels_bisect_emd_plus_pfi = bisecting_kmeans_emd(
        emd_plus_pfi_matrix, n_clusters=best_k, random_state=42
    )
    labels_bisect_emd_plus_rf = bisecting_kmeans_emd(
        emd_plus_rf_matrix, n_clusters=best_k, random_state=42
    )
    labels_bisect_emd_plus_xgb = bisecting_kmeans_emd(
        emd_plus_xgb_matrix, n_clusters=best_k, random_state=42
    )
    labels_bisect_emd_plus_shap = bisecting_kmeans_emd(
        emd_plus_shap_matrix, n_clusters=best_k, random_state=42
    )
    labels_bisect_emd_plus_lime = bisecting_kmeans_emd(
        emd_plus_lime_matrix, n_clusters=best_k, random_state=42
    )
    print("Bisecting K-Means with EMD+ completed.\n")

    K_euclid = compute_rbf_kernel(X, gamma=0.1)
    labels_kernel_euclid = kernel_kmeans(
        K_euclid, n_clusters=best_k, max_iter=100, random_state=42
    )
    print("Kernel K-Means with Euclidean completed.\n")

    K_emd = compute_rbf_kernel_from_distances(emd_matrix, gamma=0.1)
    labels_kernel_emd = kernel_kmeans(
        K_emd, n_clusters=best_k, max_iter=100, random_state=42
    )
    print("Kernel K-Means with EMD completed.\n")

    K_emd_plus_pca = compute_rbf_kernel_from_distances(emd_plus_pca_matrix, gamma=0.1)
    K_emd_plus_pfi = compute_rbf_kernel_from_distances(emd_plus_pfi_matrix, gamma=0.1)
    K_emd_plus_rf = compute_rbf_kernel_from_distances(emd_plus_rf_matrix, gamma=0.1)
    K_emd_plus_xgb = compute_rbf_kernel_from_distances(emd_plus_xgb_matrix, gamma=0.1)
    K_emd_plus_shap = compute_rbf_kernel_from_distances(emd_plus_shap_matrix, gamma=0.1)
    K_emd_plus_lime = compute_rbf_kernel_from_distances(emd_plus_lime_matrix, gamma=0.1)
    labels_kernel_emd_plus_pca = kernel_kmeans(
        K_emd_plus_pca, n_clusters=best_k, max_iter=100, random_state=42
    )
    labels_kernel_emd_plus_pfi = kernel_kmeans(
        K_emd_plus_pfi, n_clusters=best_k, max_iter=100, random_state=42
    )
    labels_kernel_emd_plus_rf = kernel_kmeans(
        K_emd_plus_rf, n_clusters=best_k, max_iter=100, random_state=42
    )
    labels_kernel_emd_plus_xgb = kernel_kmeans(
        K_emd_plus_xgb, n_clusters=best_k, max_iter=100, random_state=42
    )
    labels_kernel_emd_plus_shap = kernel_kmeans(
        K_emd_plus_shap, n_clusters=best_k, max_iter=100, random_state=42
    )
    labels_kernel_emd_plus_lime = kernel_kmeans(
        K_emd_plus_lime, n_clusters=best_k, max_iter=100, random_state=42
    )
    print("Kernel K-Means with EMD+ completed.\n")

    all_labels = {
        "K-Means (Euclidean)": labels_euclid,
        "K-Means (EMD)": labels_emd_kmeans,
        "K-Means (EMD+ PCA)": labels_emd_kmeans_plus_pca,
        "K-Means (EMD+ PFI)": labels_emd_kmeans_plus_pfi,
        "K-Means (EMD+ RF)": labels_emd_kmeans_plus_rf,
        "K-Means (EMD+ XGB)": labels_emd_kmeans_plus_xgb,
        "K-Means (EMD+ SHAP)": labels_emd_kmeans_plus_shap,
        "K-Means (EMD+ LIME)": labels_emd_kmeans_plus_lime,
        "K-Medoids (Euclidean)": labels_euclid_medoids,
        "K-Medoids (EMD)": labels_emd_medoids,
        "K-Medoids (EMD+ PCA)": labels_emd_plus_pca_medoids,
        "K-Medoids (EMD+ PFI)": labels_emd_plus_pfi_medoids,
        "K-Medoids (EMD+ RF)": labels_emd_plus_rf_medoids,
        "K-Medoids (EMD+ XGB)": labels_emd_plus_xgb_medoids,
        "K-Medoids (EMD+ SHAP)": labels_emd_plus_shap_medoids,
        "K-Medoids (EMD+ LIME)": labels_emd_plus_lime_medoids,
        "K-Means++ (Euclidean)": labels_kpp_euclid,
        "K-Means++ (EMD)": labels_kpp_emd,
        "K-Means++ (EMD+ PCA)": labels_kpp_emd_plus_pca,
        "K-Means++ (EMD+ PFI)": labels_kpp_emd_plus_pfi,
        "K-Means++ (EMD+ RF)": labels_kpp_emd_plus_rf,
        "K-Means++ (EMD+ XGB)": labels_kpp_emd_plus_xgb,
        "K-Means++ (EMD+ SHAP)": labels_kpp_emd_plus_shap,
        "K-Means++ (EMD+ LIME)": labels_kpp_emd_plus_lime,
        "Mini-Batch K-Means (Euclidean)": labels_mbk_euclid,
        "Mini-Batch K-Means (EMD)": labels_mbk_emd,
        "Mini-Batch K-Means (EMD+ PCA)": labels_mbk_emd_plus_pca,
        "Mini-Batch K-Means (EMD+ PFI)": labels_mbk_emd_plus_pfi,
        "Mini-Batch K-Means (EMD+ RF)": labels_mbk_emd_plus_rf,
        "Mini-Batch K-Means (EMD+ XGB)": labels_mbk_emd_plus_xgb,
        "Mini-Batch K-Means (EMD+ SHAP)": labels_mbk_emd_plus_shap,
        "Mini-Batch K-Means (EMD+ LIME)": labels_mbk_emd_plus_lime,
        "Bisecting K-Means (Euclidean)": labels_bisect_euclid,
        "Bisecting K-Means (EMD)": labels_bisect_emd,
        "Bisecting K-Means (EMD+ PCA)": labels_bisect_emd_plus_pca,
        "Bisecting K-Means (EMD+ PFI)": labels_bisect_emd_plus_pfi,
        "Bisecting K-Means (EMD+ RF)": labels_bisect_emd_plus_rf,
        "Bisecting K-Means (EMD+ XGB)": labels_bisect_emd_plus_xgb,
        "Bisecting K-Means (EMD+ SHAP)": labels_bisect_emd_plus_shap,
        "Bisecting K-Means (EMD+ LIME)": labels_bisect_emd_plus_lime,
        "Kernel K-Means (Euclidean)": labels_kernel_euclid,
        "Kernel K-Means (EMD)": labels_kernel_emd,
        "Kernel K-Means (EMD+ PCA)": labels_kernel_emd_plus_pca,
        "Kernel K-Means (EMD+ PFI)": labels_kernel_emd_plus_pfi,
        "Kernel K-Means (EMD+ RF)": labels_kernel_emd_plus_rf,
        "Kernel K-Means (EMD+ XGB)": labels_kernel_emd_plus_xgb,
        "Kernel K-Means (EMD+ SHAP)": labels_kernel_emd_plus_shap,
        "Kernel K-Means (EMD+ LIME)": labels_kernel_emd_plus_lime,
    }

    print("\n== Clustering Metrics for All Methods ==")
    df_metrics = print_clustering_metrics_all_methods(
        X=X,
        true_labels=(df_original[value_col].values if value_col else None),
        all_labels=all_labels,
        emd_matrix=emd_matrix,
        save_path="results/clustering_metrics_summary.csv",
    )

    excel_book = "results/clustering_metrics.xlsx"
    write_metrics_to_excel(
        df_metrics=df_metrics,
        dataset_name=Path(file_path).stem,
        k_value=best_k,
        excel_path=excel_book,
    )
    print(f"Metrics appended to '{excel_book}' (views + Overview updated).")
    print("Done.")


if __name__ == "__main__":
    main(
        file_path="data/Dataset Heart Disease.xlsx",
        value_col="target",
        id_col="Unnamed: 0",
    )

    main(file_path="data/k33 from Kaggle.xlsx", value_col="HeartDisease", id_col="ID")

    main(file_path="data/heart.xlsx", value_col="Heart_Disease")

    main(
        file_path="data/heart_attack_prediction_dataset.xlsx",
        value_col="Heart Attack Risk",
        id_col="Patient ID",
    )

    main(file_path="data/heart_disease_dataset.xlsx", value_col="Heart Disease")
