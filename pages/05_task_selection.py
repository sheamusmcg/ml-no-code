import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from components.state_manager import (
    is_data_loaded, get_working_data, clear_downstream, is_data_processed,
)

# ── Prerequisite check ─────────────────────────────────────────────────────
if not is_data_loaded():
    st.title("Choose Your Task")
    st.warning("Please load a dataset first.")
    st.page_link("pages/02_data_loading.py", label="Go to Load Data", icon=":material/arrow_back:")
    st.stop()

df = get_working_data()

st.title("Choose Your Task")
st.write(
    "Machine learning problems generally fall into three categories. "
    "Pick the one that matches what you want to do with your data."
)

if not is_data_processed():
    st.info(
        "You haven't applied any preprocessing yet. That's fine — you can always come back "
        "to the Preprocess page later. We'll use the raw data for now."
    )

# ── Task Type Selection ───────────────────────────────────────────────────
st.header("1. Select Task Type")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Classification")
    st.write("**Predict a category**")
    st.write("Examples: spam vs. not spam, species of flower, disease diagnosis")
    classify = st.button("Select Classification", key="btn_classify", use_container_width=True)

with col2:
    st.subheader("Regression")
    st.write("**Predict a number**")
    st.write("Examples: house price, temperature, sales forecast")
    regress = st.button("Select Regression", key="btn_regress", use_container_width=True)

with col3:
    st.subheader("Clustering")
    st.write("**Find groups**")
    st.write("Examples: customer segments, document topics, anomaly detection")
    cluster = st.button("Select Clustering", key="btn_cluster", use_container_width=True)

# Handle button clicks
if classify:
    st.session_state["task_type"] = "classification"
if regress:
    st.session_state["task_type"] = "regression"
if cluster:
    st.session_state["task_type"] = "clustering"

task_type = st.session_state.get("task_type")

if task_type is None:
    st.info("Select a task type above to continue.")
    st.stop()

st.success(f"Selected task: **{task_type.title()}**")

with st.expander("Learn more: Classification vs Regression vs Clustering"):
    st.write(
        "**Classification** predicts which group something belongs to (the output is a label). "
        "**Regression** predicts a continuous number (the output is a value on a number line). "
        "**Clustering** groups similar items together without any pre-defined labels "
        "(this is called *unsupervised learning* because there is no 'right answer' to learn from)."
    )

# ── Target and Feature Selection ──────────────────────────────────────────
if task_type in ("classification", "regression"):
    st.header("2. Select Target and Features")

    st.write(
        "The **target** is the column you want to predict. "
        "The **features** are the columns the model uses to make that prediction."
    )

    # Target column
    all_cols = df.columns.tolist()
    default_target_idx = len(all_cols) - 1  # Default to last column

    target = st.selectbox(
        "Target column (what to predict)",
        all_cols,
        index=default_target_idx,
        help="Choose the column that contains the values you want to predict.",
    )

    # Suggest task type mismatch
    n_unique = df[target].nunique()
    if task_type == "classification" and n_unique > 20:
        st.warning(
            f"This column has {n_unique} unique values. That's a lot for classification. "
            "Are you sure this isn't a regression problem?"
        )
    elif task_type == "regression" and n_unique <= 5:
        st.warning(
            f"This column has only {n_unique} unique values. This might be better suited "
            "for classification."
        )

    # Feature columns
    available_features = [c for c in all_cols if c != target]
    features = st.multiselect(
        "Feature columns (inputs to the model)",
        available_features,
        default=available_features,
        help="Select which columns the model should use as inputs.",
    )

    if not features:
        st.error("Please select at least one feature column.")
        st.stop()

    st.session_state["target_column"] = target
    st.session_state["feature_columns"] = features

elif task_type == "clustering":
    st.header("2. Select Features")
    st.write(
        "Clustering doesn't have a target column — it discovers groups on its own. "
        "Select the features you want the algorithm to use for grouping."
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Clustering requires numeric features. Please go back and encode your categorical columns.")
        st.stop()

    features = st.multiselect(
        "Feature columns",
        numeric_cols,
        default=numeric_cols,
        help="Select numeric columns for clustering.",
    )

    if not features:
        st.error("Please select at least one feature column.")
        st.stop()

    st.session_state["target_column"] = None
    st.session_state["feature_columns"] = features

# ── Train/Test Split ──────────────────────────────────────────────────────
st.header("3. Split Data into Training and Testing Sets")

with st.expander("Learn more: Why split the data?"):
    st.write(
        "Imagine studying for an exam using a practice test. If you memorize the exact answers, "
        "you might score perfectly on that practice test — but that doesn't mean you understand "
        "the material. In the same way, we hold out some data (the **test set**) that the model "
        "never sees during training. We use this to check if the model truly learned the pattern, "
        "or just memorized the training data."
    )

col_a, col_b = st.columns(2)

with col_a:
    test_size = st.slider(
        "Test set size (%)",
        10, 50, 20, step=5,
        help="What fraction of data to hold out for testing. 20% is a common choice.",
    )

with col_b:
    random_seed = st.number_input(
        "Random seed",
        min_value=0, max_value=99999, value=42,
        help="A fixed seed makes the split reproducible. Use the same seed to get the same split every time.",
    )

stratify_option = False
if task_type == "classification":
    stratify_option = st.checkbox(
        "Stratified split",
        value=True,
        help="Ensures both training and test sets have the same class proportions as the full dataset.",
    )

# ── Perform Split ─────────────────────────────────────────────────────────
if st.button("Prepare Data", type="primary"):
    try:
        X = df[features].copy()
        # Ensure all features are numeric for modeling
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            st.error(
                f"The following feature columns are not numeric: {', '.join(non_numeric)}. "
                "Please go back to the Preprocess page and encode them."
            )
            st.stop()

        X_values = X.values

        if task_type in ("classification", "regression"):
            y = df[st.session_state["target_column"]].copy()
            # Encode string targets for classification
            if task_type == "classification" and y.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                st.session_state["label_encoder"] = le
                st.session_state["target_names"] = le.classes_.tolist()
            else:
                y_encoded = y.values
                st.session_state["label_encoder"] = None
                if task_type == "classification":
                    st.session_state["target_names"] = sorted(y.unique().tolist())
                else:
                    st.session_state["target_names"] = None

            stratify = y_encoded if stratify_option else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_values, y_encoded,
                test_size=test_size / 100,
                random_state=int(random_seed),
                stratify=stratify,
            )
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
        else:
            # Clustering: no target, but still split for consistency
            X_train, X_test = train_test_split(
                X_values,
                test_size=test_size / 100,
                random_state=int(random_seed),
            )
            st.session_state["y_train"] = None
            st.session_state["y_test"] = None

        clear_downstream("split")
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["test_size"] = test_size / 100
        st.session_state["random_seed"] = int(random_seed)

        st.success(
            f"Data split complete! Training set: {X_train.shape[0]:,} samples, "
            f"Test set: {X_test.shape[0]:,} samples, "
            f"Features: {X_train.shape[1]}."
        )
    except Exception as e:
        st.error(f"Error during data split: {e}")

# ── Navigation ─────────────────────────────────────────────────────────────
st.divider()
if st.session_state.get("X_train") is not None:
    st.page_link("pages/06_model_training.py", label="Next: Train a Model", icon=":material/arrow_forward:")
