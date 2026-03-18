import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from components.state_manager import (
    is_data_loaded, get_working_data, clear_downstream, is_data_processed,
)
from components.preprocessing_utils import encode_categoricals, handle_missing_values

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

# ── Auto-handle non-numeric features ──────────────────────────────────────
X_preview = df[features].copy()
non_numeric = X_preview.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric:
    st.header("Auto-Encode Categorical Features")
    st.warning(
        f"**{len(non_numeric)} categorical (text) column(s) detected** in your features. "
        "ML models need numbers, not text. Choose how to convert them automatically."
    )

    with st.expander(f"Show categorical columns ({len(non_numeric)})"):
        st.write(", ".join(non_numeric))

    auto_encode = st.radio(
        "How should we encode these columns?",
        ["label", "onehot", "drop"],
        format_func=lambda x: {
            "label": "Label Encoding — assign a number to each category (fast, compact)",
            "onehot": "One-Hot Encoding — create 0/1 columns per category (more accurate, more columns)",
            "drop": "Drop them — remove all categorical columns",
        }[x],
        help="Label encoding is simpler but implies an order. One-hot encoding is more accurate but creates many columns.",
    )

    with st.expander("Learn more: Label vs One-Hot Encoding"):
        st.write(
            "**Label Encoding** converts each unique category to a number. For example, "
            "'Red'=0, 'Blue'=1, 'Green'=2. This is fast and compact, but the model might "
            "think Green(2) > Blue(1) > Red(0), which may not be meaningful.\n\n"
            "**One-Hot Encoding** creates a new column for each unique category. For example, "
            "'Color_Red', 'Color_Blue', 'Color_Green', each with 0 or 1. This avoids the "
            "ordering problem but can create lots of columns with high-cardinality features.\n\n"
            "**Drop** simply removes all text columns. Use this if the categorical columns "
            "aren't useful for prediction."
        )

# ── Handle missing values automatically ───────────────────────────────────
has_missing = df[features].isnull().sum().sum() > 0
if has_missing:
    missing_count = df[features].isnull().sum().sum()
    missing_cols = df[features].isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    st.header("Auto-Handle Missing Values")
    st.warning(
        f"**{missing_count:,} missing value(s)** found across {len(missing_cols)} column(s). "
        "These will be handled automatically before training."
    )
    auto_missing = st.radio(
        "How should we handle missing values?",
        ["drop_rows", "fill_median_mode"],
        format_func=lambda x: {
            "drop_rows": "Drop rows with any missing values",
            "fill_median_mode": "Fill with median (numbers) / mode (categories)",
        }[x],
        help="Dropping rows is simple but loses data. Filling preserves all rows.",
    )

# ── Perform Split ─────────────────────────────────────────────────────────
if st.button("Prepare Data", type="primary"):
    try:
        work_df = df[features].copy()

        # Step 1: Handle missing values
        if has_missing:
            if auto_missing == "drop_rows":
                before = len(work_df)
                work_df = work_df.dropna()
                dropped = before - len(work_df)
                st.info(f"Dropped {dropped:,} rows with missing values ({len(work_df):,} remaining).")
                # Also drop from the full df to keep target aligned
                df = df.loc[work_df.index]
            else:
                # Fill numeric with median, categorical with mode
                for col in work_df.columns:
                    if work_df[col].isnull().sum() > 0:
                        if work_df[col].dtype in [np.float64, np.int64, float, int]:
                            work_df[col] = work_df[col].fillna(work_df[col].median())
                        else:
                            mode_val = work_df[col].mode()
                            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                            work_df[col] = work_df[col].fillna(fill_val)

        # Step 2: Encode categoricals
        current_non_numeric = work_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if current_non_numeric:
            if auto_encode == "label":
                for col in current_non_numeric:
                    le_feat = LabelEncoder()
                    work_df[col] = le_feat.fit_transform(work_df[col].astype(str))
                st.info(f"Label-encoded {len(current_non_numeric)} categorical columns.")
            elif auto_encode == "onehot":
                before_cols = len(work_df.columns)
                work_df = pd.get_dummies(work_df, columns=current_non_numeric, drop_first=True, dtype=int)
                st.info(f"One-hot encoded: {before_cols} columns -> {len(work_df.columns)} columns.")
            elif auto_encode == "drop":
                work_df = work_df.drop(columns=current_non_numeric)
                st.info(f"Dropped {len(current_non_numeric)} categorical columns. {len(work_df.columns)} features remaining.")

        if work_df.shape[1] == 0:
            st.error("No features remaining after encoding. Please select different features.")
            st.stop()

        X_values = work_df.values
        # Store the final feature names (may have changed with one-hot encoding)
        final_feature_names = work_df.columns.tolist()

        if task_type in ("classification", "regression"):
            y = df.loc[work_df.index, st.session_state["target_column"]].copy()
            # Encode string targets for classification
            if task_type == "classification" and y.dtype == object:
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
        st.session_state["feature_columns"] = final_feature_names
        st.session_state["test_size"] = test_size / 100
        st.session_state["random_seed"] = int(random_seed)

        st.success(
            f"Data prepared! Training set: {X_train.shape[0]:,} samples, "
            f"Test set: {X_test.shape[0]:,} samples, "
            f"Features: {X_train.shape[1]}."
        )
    except Exception as e:
        st.error(f"Error during data preparation: {e}")

# ── Navigation ─────────────────────────────────────────────────────────────
st.divider()
if st.session_state.get("X_train") is not None:
    st.page_link("pages/06_model_training.py", label="Next: Train a Model", icon=":material/arrow_forward:")
