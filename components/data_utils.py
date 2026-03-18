import pandas as pd
import numpy as np
import streamlit as st
from sklearn import datasets


BUILTIN_DATASETS = {
    "Classification": {
        "Iris": {
            "loader": lambda: datasets.load_iris(as_frame=True),
            "description": (
                "150 samples of iris flowers with 4 measurements (sepal length/width, "
                "petal length/width). The goal is to predict which of 3 species a flower belongs to. "
                "A classic beginner dataset."
            ),
        },
        "Wine": {
            "loader": lambda: datasets.load_wine(as_frame=True),
            "description": (
                "178 samples of wine from 3 different cultivars, described by 13 chemical "
                "properties (alcohol, color intensity, etc.). Predict which cultivar produced the wine."
            ),
        },
        "Breast Cancer": {
            "loader": lambda: datasets.load_breast_cancer(as_frame=True),
            "description": (
                "569 samples of breast tissue measurements. Each sample has 30 numeric features "
                "computed from cell images. The goal is to predict whether a tumor is malignant or benign."
            ),
        },
        "Digits": {
            "loader": lambda: datasets.load_digits(as_frame=True),
            "description": (
                "1,797 small (8x8 pixel) images of handwritten digits (0-9). Each pixel is a feature "
                "(64 features total). Predict which digit is shown."
            ),
        },
    },
    "Regression": {
        "Diabetes": {
            "loader": lambda: datasets.load_diabetes(as_frame=True),
            "description": (
                "442 diabetes patients with 10 baseline measurements (age, BMI, blood pressure, etc.). "
                "The target is a measure of disease progression one year later."
            ),
        },
        "California Housing": {
            "loader": lambda: _load_california_housing(),
            "description": (
                "20,640 California census block groups. Features include median income, house age, "
                "and location. The target is the median house value (in $100,000s)."
            ),
        },
    },
    "Clustering": {
        "Blobs": {
            "loader": lambda: _make_blobs(),
            "description": (
                "Synthetic data with clear, separated groups (clusters). Good for learning how "
                "clustering algorithms find natural groupings in data."
            ),
            "configurable": True,
        },
        "Moons": {
            "loader": lambda: _make_moons(),
            "description": (
                "Two interleaving half-moon shapes. Tests whether a clustering algorithm can find "
                "non-spherical clusters."
            ),
        },
        "Circles": {
            "loader": lambda: _make_circles(),
            "description": (
                "Two concentric circles. Tests whether a clustering algorithm can find nested clusters "
                "that aren't linearly separable."
            ),
        },
    },
}


@st.cache_data
def _load_california_housing():
    return datasets.fetch_california_housing(as_frame=True)


def _make_blobs(n_samples=300, n_clusters=3, n_features=2, random_state=42):
    X, y = datasets.make_blobs(
        n_samples=n_samples, centers=n_clusters,
        n_features=n_features, random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
    df["Cluster"] = y
    return df


def _make_moons(n_samples=300, noise=0.1, random_state=42):
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
    df["Cluster"] = y
    return df


def _make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42):
    X, y = datasets.make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=random_state,
    )
    df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
    df["Cluster"] = y
    return df


def load_builtin_dataset(category: str, name: str, **kwargs) -> pd.DataFrame:
    """Load a built-in dataset and return it as a DataFrame.

    For sklearn Bunch objects, combines features and target into one DataFrame.
    For synthetic datasets, returns the DataFrame directly.
    """
    entry = BUILTIN_DATASETS[category][name]
    result = entry["loader"]()

    # Synthetic datasets return a DataFrame directly
    if isinstance(result, pd.DataFrame):
        return result

    # sklearn Bunch object
    bunch = result
    df = bunch.frame.copy() if hasattr(bunch, "frame") and bunch.frame is not None else pd.DataFrame(
        bunch.data, columns=bunch.feature_names
    )
    if "target" not in df.columns and hasattr(bunch, "target"):
        target = bunch.target
        if hasattr(bunch, "target_names") and bunch.target_names is not None:
            # Map numeric targets to string labels for classification
            if category == "Classification":
                target = pd.Series(target).map(
                    dict(enumerate(bunch.target_names))
                ).values
        df["target"] = target
    return df


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load and validate an uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    if df.shape[0] < 2:
        raise ValueError("The CSV file must contain at least 2 rows.")
    if df.shape[1] < 2:
        raise ValueError("The CSV file must contain at least 2 columns.")
    return df


def get_column_types(df: pd.DataFrame) -> dict:
    """Categorize columns as numeric or categorical."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return {"numeric": numeric, "categorical": categorical}
