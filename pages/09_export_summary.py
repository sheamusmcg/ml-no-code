import io
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from components.state_manager import is_data_loaded, has_trained_models
from components.evaluation_utils import (
    classification_metrics, regression_metrics, clustering_metrics,
)

st.title("Export & Summary")
st.write("Review the full pipeline you followed and download your results.")

# ── Pipeline Summary ──────────────────────────────────────────────────────
st.header("Pipeline Summary")

steps = []

# 1. Data
if is_data_loaded():
    name = st.session_state.get("data_name", "Unknown")
    shape = st.session_state["data_raw"].shape
    steps.append(f"Loaded dataset **{name}** ({shape[0]:,} rows, {shape[1]} columns)")
else:
    st.info("No dataset loaded yet. Start from the Load Data page.")
    st.page_link("pages/02_data_loading.py", label="Go to Load Data", icon=":material/arrow_back:")
    st.stop()

# 2. Preprocessing
preproc_steps = st.session_state.get("preprocessing_steps", [])
if preproc_steps:
    steps.append("**Preprocessing applied:**")
    for ps in preproc_steps:
        steps.append(f"  - {ps}")
else:
    steps.append("No preprocessing applied (used raw data)")

# 3. Task
task_type = st.session_state.get("task_type")
if task_type:
    target = st.session_state.get("target_column")
    features = st.session_state.get("feature_columns", [])
    if target:
        steps.append(f"Task: **{task_type.title()}** — predicting '{target}' using {len(features)} features")
    else:
        steps.append(f"Task: **{task_type.title()}** using {len(features)} features")

# 4. Split
if st.session_state.get("X_train") is not None:
    train_n = st.session_state["X_train"].shape[0]
    test_n = st.session_state["X_test"].shape[0]
    steps.append(f"Data split: {train_n:,} training / {test_n:,} testing samples")

# 5. Models
trained = st.session_state.get("trained_models", {})
if trained:
    steps.append(f"**{len(trained)} model(s) trained:**")
    for name, result in trained.items():
        steps.append(f"  - {name} ({result['model_type']}, {result['train_time']:.2f}s)")

for step in steps:
    st.write(step)

# ── Downloads ─────────────────────────────────────────────────────────────
if not has_trained_models():
    st.info("Train a model to unlock download options.")
    st.stop()

st.header("Download Results")

y_test = st.session_state.get("y_test")
X_test = st.session_state.get("X_test")

for model_name, result in trained.items():
    st.subheader(model_name)
    col1, col2, col3 = st.columns(3)

    # Predictions CSV
    with col1:
        pred_data = {"prediction": result["y_pred_test"]}
        if y_test is not None:
            pred_data["actual"] = y_test
        pred_df = pd.DataFrame(pred_data)
        csv_buffer = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_buffer,
            file_name=f"{model_name.replace(' ', '_')}_predictions.csv",
            mime="text/csv",
            key=f"dl_pred_{model_name}",
        )

    # Metrics CSV
    with col2:
        if task_type == "classification":
            m = classification_metrics(y_test, result["y_pred_test"], result.get("y_prob_test"))
        elif task_type == "regression":
            m = regression_metrics(y_test, result["y_pred_test"])
        elif task_type == "clustering":
            m = clustering_metrics(st.session_state["X_train"], result["y_pred_train"])
        else:
            m = {}

        metrics_df = pd.DataFrame([m])
        csv_metrics = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics (CSV)",
            data=csv_metrics,
            file_name=f"{model_name.replace(' ', '_')}_metrics.csv",
            mime="text/csv",
            key=f"dl_metrics_{model_name}",
        )

    # Model pickle
    with col3:
        model_bytes = io.BytesIO()
        pickle.dump(result["model"], model_bytes)
        model_bytes.seek(0)
        st.download_button(
            label="Download Model (pickle)",
            data=model_bytes,
            file_name=f"{model_name.replace(' ', '_')}_model.pkl",
            mime="application/octet-stream",
            key=f"dl_model_{model_name}",
        )

st.caption(
    "**Note:** Pickle files can execute arbitrary code when loaded. "
    "Only load pickle files from sources you trust."
)

# ── Try Your Model ────────────────────────────────────────────────────────
st.header("Try Your Model on New Data")
st.write(
    "Ready to see your model in action? The **Predict** page lets you input new values "
    "manually or upload a fresh CSV — just like deploying a model in the real world."
)
st.page_link("pages/10_predict.py", label="Go to Predict", icon=":material/rocket_launch:")

# ── What's Next ───────────────────────────────────────────────────────────
st.header("What's Next?")
st.write(
    "Congratulations on completing the ML pipeline! Here are some ideas for next steps:\n\n"
    "- **Try your model** — go to the Predict page and test it on new data\n"
    "- **Try different models** — go back to the Training page and experiment with other algorithms\n"
    "- **Tune hyperparameters** — adjust the settings and see how performance changes\n"
    "- **Try different preprocessing** — does scaling help? Does removing features improve results?\n"
    "- **Upload your own data** — apply what you learned to a real-world dataset\n"
    "- **Learn more** — explore the scikit-learn documentation at scikit-learn.org"
)
