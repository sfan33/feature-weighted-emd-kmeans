import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

def make_target_binary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    unique_vals = df[label_col].dropna().unique()
    if len(unique_vals) != 2:
        raise ValueError(
            f"Target column '{label_col}' must have exactly 2 unique values, "
            f"but found {len(unique_vals)}: {list(unique_vals)}"
        )
    
    # Map the first unique value to 0, the second to 1
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    df[label_col] = df[label_col].map(mapping)
    df[label_col] = df[label_col].astype(int)
    
    return df

def preprocess_for_pairwise(df: pd.DataFrame, encoding_method: str = None, label_col: str = None ) -> pd.DataFrame:
    df = df.copy()

    target_series = None
    if label_col and label_col in df.columns:
        if not np.issubdtype(df[label_col].dtype, np.number):
            df = make_target_binary(df, label_col)
        target_series = df[label_col].copy()
        df.drop(columns=[label_col], inplace=True)
    
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if encoding_method.lower() == "onehot":
        cat_encoded = []
        for col in cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col) # Turns each category into its own binary column
            cat_encoded.append(dummies)
        if cat_encoded:
            cat_part = pd.concat(cat_encoded, axis=1)
        else:
            cat_part = pd.DataFrame(index=df.index)
    elif encoding_method.lower() == "target":
        cat_part = pd.DataFrame(index=df.index)
        for col in cat_cols:
            means = target_series.groupby(df[col]).mean()
            cat_part[col + "_target"] = df[col].map(means)
    else:
        cat_part = pd.DataFrame(index=df.index)
        for col in cat_cols:
            uniques = sorted(df[col].unique())
            mapping_dict = {cat_val: i + 1 for i, cat_val in enumerate(uniques)}
            cat_part[col + "_ord"] = df[col].map(mapping_dict)
    
    num_part = df[num_cols].copy()
    df_processed = pd.concat([num_part, cat_part], axis=1)

    df_processed.dropna(inplace=True)
    
    col_max = df_processed.max()
    num_columns = df_processed.shape[1]
    df_normalized = df_processed.div(col_max * num_columns, axis=1)
    row_sums = df_normalized.sum(axis=1)
    df_normalized['Adjusted_p'] = 1 - row_sums
    df_normalized.to_csv(f"results/data_after_normalization_{encoding_method}.csv", index=False)

    return target_series, df_normalized
