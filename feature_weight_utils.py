import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import lime.lime_tabular

def get_rf_weights(df, label_col, random_state=42):
    """
    Compute feature importances using Random Forest.
    """
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X, y)

    importances = model.feature_importances_
    weights = importances / np.sum(importances)

    return weights

def get_xgb_weights(df, label_col, random_state=42):
    """
    Compute feature importances using XGBoost.
    """
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()

    model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)

    # Get raw importances from booster (can use gain, cover, or weight)
    booster = model.get_booster()
    fmap = booster.get_fscore()
    
    # Map feature importances to the order of columns
    feature_names = X.columns.tolist()
    importances = np.array([fmap.get(f"f{i}", 0) for i in range(len(feature_names))])
    weights = importances / np.sum(importances) if importances.sum() != 0 else np.ones(len(importances)) / len(importances)

    return weights

def get_shap_weights(df_with_label, label_col, random_state=42):
    X = df_with_label.drop(columns=[label_col])
    y = df_with_label[label_col]

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        sv = np.stack(sv, axis=0)
        sv = np.abs(sv).mean(axis=0)
    else:
        sv = np.abs(sv)

    importances = sv.mean(axis=0)

    if importances.ndim > 1:
        importances = importances.mean(axis=1)
    weights = importances / importances.sum()
    return weights

def get_lime_weights(df, label_col, num_samples=100, random_state=42):
    """
    Compute global LIME feature importances by averaging over multiple samples.
    """
    X_df = df.drop(columns=[label_col]).reset_index(drop=True)
    y = df[label_col].values
    X = X_df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_scaled, y)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data    = X_scaled,
        feature_names    = X_df.columns.tolist(),
        class_names      = [str(c) for c in np.unique(y)],
        mode             = "classification",
        discretize_continuous = True,
        random_state     = random_state
    )

    importances = np.zeros(X.shape[1], dtype=float)
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(len(X), size=min(num_samples, len(X)), replace=False)

    for idx in sample_indices:
        exp = explainer.explain_instance(
            X_scaled[idx],
            model.predict_proba,
            num_features=X.shape[1]
        )
        for feature_str, weight in exp.as_list():
            if " <= " in feature_str:
                base_name = feature_str.split(" <= ")[0]
            elif " > " in feature_str:
                base_name = feature_str.split(" > ")[0]
            else:
                base_name = feature_str

            if base_name in X_df.columns:
                col_idx = X_df.columns.get_loc(base_name)
                importances[col_idx] += abs(weight)

    weights = importances / importances.sum()
    return weights
