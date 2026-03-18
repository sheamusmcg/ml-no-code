import streamlit as st
import pandas as pd
import numpy as np
from components.state_manager import is_data_loaded, clear_downstream
from components.data_utils import get_column_types
from components.preprocessing_utils import (
    handle_missing_values,
    scale_features,
    encode_categoricals,
    apply_variance_threshold,
)

# ── Prerequisite check ─────────────────────────────────────────────────────
if not is_data_loaded():
    st.title("Preprocess Data")
    st.warning("Please load a dataset first.")
    st.page_link("pages/02_data_loading.py", label="Go to Load Data", icon=":material/arrow_back:")
    st.stop()

df = st.session_state["data_raw"].copy()
col_types = get_column_types(df)

st.title("Preprocess Data")
st.write(
    "Real-world data is rarely perfect. This page lets you clean and transform your data "
    "so that machine learning models can work with it effectively. Each step below is optional "
    "— apply only what makes sense for your data."
)

all_steps = []

# ── 1. Handle Missing Values ──────────────────────────────────────────────
st.header("1. Handle Missing Values")

missing = df.isnull().sum()
cols_with_missing = missing[missing > 0]

if len(cols_with_missing) == 0:
    st.success("No missing values found. You can skip this step.")
else:
    st.write(f"Found missing values in **{len(cols_with_missing)}** column(s).")

    with st.expander("Learn more: Why handle missing values?"):
        st.write(
            "Most machine learning models cannot work with missing (blank) values. "
            "You can either **drop** the rows that have missing values, or **fill** them in "
            "with a reasonable guess like the average (mean), middle value (median), or most "
            "common value (mode)."
        )

    missing_strategies = {}
    for col_name in cols_with_missing.index:
        count = cols_with_missing[col_name]
        pct = count / len(df) * 100

        is_numeric = col_name in col_types["numeric"]
        options = ["Keep as-is", "Drop rows"]
        if is_numeric:
            options += ["Fill with mean", "Fill with median"]
        options.append("Fill with mode")

        strategy = st.selectbox(
            f"**{col_name}** — {count} missing ({pct:.1f}%)",
            options,
            help=f"Choose how to handle the {count} missing values in this column.",
            key=f"missing_{col_name}",
        )

        strategy_map = {
            "Drop rows": "drop",
            "Fill with mean": "mean",
            "Fill with median": "median",
            "Fill with mode": "mode",
        }
        if strategy in strategy_map:
            missing_strategies[col_name] = strategy_map[strategy]

    if missing_strategies:
        df, steps = handle_missing_values(df, missing_strategies)
        all_steps.extend(steps)

# ── 2. Feature Scaling ────────────────────────────────────────────────────
st.header("2. Feature Scaling")

with st.expander("Learn more: Why scale features?"):
    st.write(
        "Features often have very different ranges. For example, age might go from 0 to 100, "
        "while income goes from 0 to 1,000,000. Some algorithms (like KNN and SVM) are sensitive "
        "to these differences. Scaling puts all features on a similar range so no single feature "
        "dominates just because it has bigger numbers."
    )

scaling_method = st.radio(
    "Scaling method",
    ["none", "standard", "minmax", "robust"],
    format_func=lambda x: {
        "none": "None (keep original values)",
        "standard": "Standard Scaler (mean=0, std=1)",
        "minmax": "Min-Max Scaler (scale to 0-1)",
        "robust": "Robust Scaler (uses median; good with outliers)",
    }[x],
    help="Choose a method to scale your numeric features.",
)

# Identify numeric columns that are likely features (not the target)
numeric_features = col_types["numeric"]

if scaling_method != "none" and numeric_features:
    df, step = scale_features(df, numeric_features, scaling_method)
    if step:
        all_steps.append(step)

# ── 3. Encode Categorical Variables ───────────────────────────────────────
st.header("3. Encode Categorical Variables")

categorical_cols = col_types["categorical"]

if not categorical_cols:
    st.success("No categorical (text) columns found. You can skip this step.")
else:
    st.write(f"Found **{len(categorical_cols)}** categorical column(s): {', '.join(categorical_cols)}")

    with st.expander("Learn more: Why encode categories?"):
        st.write(
            "Machine learning models work with numbers, not text. If you have a column like "
            "'Color' with values 'Red', 'Blue', 'Green', you need to convert it to numbers. "
            "**Label Encoding** assigns each category a number (Red=0, Blue=1, Green=2). "
            "**One-Hot Encoding** creates a new column for each category with 0/1 values."
        )

    encoding_method = st.radio(
        "Encoding method",
        ["none", "label", "onehot"],
        format_func=lambda x: {
            "none": "None (keep as text)",
            "label": "Label Encoding (assign numbers)",
            "onehot": "One-Hot Encoding (create 0/1 columns)",
        }[x],
        help="Choose how to convert text columns to numbers.",
    )

    if encoding_method != "none":
        # Let user pick which columns to encode
        cols_to_encode = st.multiselect(
            "Columns to encode",
            categorical_cols,
            default=categorical_cols,
            help="Select which categorical columns to encode.",
        )
        if cols_to_encode:
            df, steps = encode_categoricals(df, cols_to_encode, encoding_method)
            all_steps.extend(steps)

# ── 4. Feature Selection ─────────────────────────────────────────────────
st.header("4. Feature Selection (Optional)")

with st.expander("Learn more: Why select features?"):
    st.write(
        "Sometimes your dataset has columns that are not useful for prediction — "
        "they might be IDs, constants, or noise. Removing them can make your model simpler "
        "and sometimes more accurate."
    )

# Manual column removal
cols_to_drop = st.multiselect(
    "Columns to remove",
    df.columns.tolist(),
    help="Select any columns you want to remove from the dataset.",
)
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    all_steps.append(f"Removed columns: {', '.join(cols_to_drop)}")

# Variance threshold
updated_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
if updated_numeric:
    var_threshold = st.slider(
        "Remove low-variance features (threshold)",
        0.0, 1.0, 0.0, step=0.01,
        help="Remove numeric columns whose variance is below this value. "
             "0 = keep all columns. Increase to remove near-constant columns.",
    )
    if var_threshold > 0:
        df, steps = apply_variance_threshold(df, updated_numeric, var_threshold)
        all_steps.extend(steps)

# ── Apply and Preview ─────────────────────────────────────────────────────
st.divider()
st.header("Preview & Apply")

left, right = st.columns(2)
with left:
    st.subheader("Original Data")
    st.write(f"{st.session_state['data_raw'].shape[0]} rows, {st.session_state['data_raw'].shape[1]} columns")
    st.dataframe(st.session_state["data_raw"].head(5), use_container_width=True)

with right:
    st.subheader("Processed Data")
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(5), use_container_width=True)

if all_steps:
    st.subheader("Steps Applied")
    for i, step in enumerate(all_steps, 1):
        st.write(f"{i}. {step}")

if st.button("Save Preprocessed Data", type="primary"):
    clear_downstream("preprocessing")
    st.session_state["data_processed"] = df
    st.session_state["preprocessing_steps"] = all_steps
    st.success(f"Saved! Processed dataset has {df.shape[0]:,} rows and {df.shape[1]} columns.")

# ── Navigation ─────────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/05_task_selection.py", label="Next: Choose Your Task", icon=":material/arrow_forward:")
