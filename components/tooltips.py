"""Short help strings for Streamlit widget `help=` parameters.

Centralized here for consistency and easy editing.
"""

DATA = {
    "dataset_category": (
        "Classification = predict a category, Regression = predict a number, "
        "Clustering = find groups."
    ),
    "csv_upload": "Comma-separated values file. Maximum 50 MB.",
    "n_samples": "How many data points to generate.",
    "n_clusters": "How many distinct groups to create.",
    "n_features": "How many measurements per data point.",
}

PREPROCESSING = {
    "missing_values": "Choose how to handle missing (blank) values in this column.",
    "scaling": "Choose a method to rescale your numeric features.",
    "encoding": "Choose how to convert text columns to numbers.",
    "columns_to_encode": "Select which categorical columns to encode.",
    "columns_to_drop": "Select any columns you want to remove from the dataset.",
    "variance_threshold": (
        "Remove numeric columns whose variance is below this value. "
        "0 = keep all columns."
    ),
}

TASK = {
    "target": "Choose the column that contains the values you want to predict.",
    "features": "Select which columns the model should use as inputs.",
    "test_size": "What fraction of data to hold out for testing (0.2 = 20%).",
    "random_seed": (
        "A fixed seed makes the split reproducible. "
        "Use the same seed to get the same split every time."
    ),
    "stratify": (
        "Ensures both training and test sets have the same class proportions "
        "as the full dataset."
    ),
}

TRAINING = {
    "model_select": "Pick an algorithm to train on your data.",
    "run_name": (
        "Give this training run a name so you can identify it later "
        "when comparing models."
    ),
}

# Hyperparameter tooltips are defined in model_registry.py alongside each model.
