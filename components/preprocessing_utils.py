import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


def handle_missing_values(df: pd.DataFrame, strategies: dict) -> tuple[pd.DataFrame, list[str]]:
    """Apply missing-value strategies per column.

    Args:
        df: Input DataFrame.
        strategies: Dict mapping column name -> strategy string.
            Strategies: "drop", "mean", "median", "mode", "constant:VALUE"

    Returns:
        (processed DataFrame, list of step descriptions)
    """
    df = df.copy()
    steps = []

    for col, strategy in strategies.items():
        if df[col].isnull().sum() == 0:
            continue

        missing_count = df[col].isnull().sum()

        if strategy == "drop":
            df = df.dropna(subset=[col])
            steps.append(f"Dropped {missing_count} rows with missing '{col}'")
        elif strategy == "mean":
            val = df[col].mean()
            df[col] = df[col].fillna(val)
            steps.append(f"Filled missing '{col}' with mean ({val:.2f})")
        elif strategy == "median":
            val = df[col].median()
            df[col] = df[col].fillna(val)
            steps.append(f"Filled missing '{col}' with median ({val:.2f})")
        elif strategy == "mode":
            val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(val)
            steps.append(f"Filled missing '{col}' with mode ({val})")
        elif strategy.startswith("constant:"):
            val = strategy.split(":", 1)[1]
            df[col] = df[col].fillna(val)
            steps.append(f"Filled missing '{col}' with constant '{val}'")

    return df, steps


def scale_features(df: pd.DataFrame, numeric_cols: list, method: str) -> tuple[pd.DataFrame, str]:
    """Scale numeric columns using the specified method.

    Returns:
        (processed DataFrame, step description)
    """
    if method == "none" or not numeric_cols:
        return df, ""

    df = df.copy()

    scalers = {
        "standard": (StandardScaler(), "StandardScaler (zero mean, unit variance)"),
        "minmax": (MinMaxScaler(), "MinMaxScaler (0 to 1 range)"),
        "robust": (RobustScaler(), "RobustScaler (uses median and IQR, robust to outliers)"),
    }

    scaler, desc = scalers[method]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, f"Scaled {len(numeric_cols)} numeric columns with {desc}"


def encode_categoricals(df: pd.DataFrame, categorical_cols: list, method: str) -> tuple[pd.DataFrame, list[str]]:
    """Encode categorical columns.

    Returns:
        (processed DataFrame, list of step descriptions)
    """
    if not categorical_cols:
        return df, []

    df = df.copy()
    steps = []

    if method == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            n_classes = len(le.classes_)
            steps.append(f"Label-encoded '{col}' ({n_classes} unique values -> integers)")
    elif method == "onehot":
        before_cols = len(df.columns)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
        after_cols = len(df.columns)
        steps.append(
            f"One-hot encoded {len(categorical_cols)} columns "
            f"({before_cols} -> {after_cols} columns)"
        )

    return df, steps


def apply_variance_threshold(df: pd.DataFrame, numeric_cols: list, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Remove numeric columns with variance below threshold.

    Returns:
        (filtered DataFrame, list of step descriptions)
    """
    if threshold <= 0 or not numeric_cols:
        return df, []

    df = df.copy()
    variances = df[numeric_cols].var()
    low_var = variances[variances < threshold].index.tolist()

    if low_var:
        df = df.drop(columns=low_var)
        return df, [f"Removed {len(low_var)} low-variance columns (threshold={threshold}): {', '.join(low_var)}"]

    return df, []
