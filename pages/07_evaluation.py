import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from components.state_manager import has_trained_models
from components.evaluation_utils import (
    classification_metrics, plot_confusion_matrix, plot_roc_curve,
    classification_report_df, plot_feature_importance, plot_learning_curve,
    regression_metrics, plot_actual_vs_predicted, plot_residuals, plot_residual_histogram,
    clustering_metrics, plot_clusters_2d, plot_elbow,
)

# ── Prerequisite check ─────────────────────────────────────────────────────
if not has_trained_models():
    st.title("Evaluate Your Model")
    st.warning("Please train a model first.")
    st.page_link("pages/06_model_training.py", label="Go to Train Model", icon=":material/arrow_back:")
    st.stop()

trained = st.session_state["trained_models"]

st.title("Evaluate Your Model")
st.write(
    "Now let's see how well your model actually performs. We test it on data it has "
    "never seen before (the test set) to get an honest estimate of its quality."
)

# ── Model Selector ────────────────────────────────────────────────────────
model_names = list(trained.keys())
selected = st.selectbox(
    "Select a model to evaluate",
    model_names,
    index=model_names.index(st.session_state.get("current_model_name", model_names[0]))
    if st.session_state.get("current_model_name") in model_names else 0,
    help="Pick which trained model to evaluate.",
)

result = trained[selected]
task_type = result["task_type"]
model = result["model"]
y_pred_test = result["y_pred_test"]
y_pred_train = result["y_pred_train"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state.get("y_train")
y_test = st.session_state.get("y_test")
feature_names = result.get("feature_columns", [])

st.info(f"**Model:** {result['model_type']} | **Task:** {task_type.title()} | **Training time:** {result['train_time']:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
if task_type == "classification":
    y_prob = result.get("y_prob_test")
    target_names = st.session_state.get("target_names")

    # ── Metrics ────────────────────────────────────────────────────────
    st.header("Metrics")
    m = classification_metrics(y_test, y_pred_test, y_prob)

    cols = st.columns(len(m))
    for i, (name, val) in enumerate(m.items()):
        cols[i].metric(name, f"{val:.4f}")

    with st.expander("Learn more: Classification metrics explained"):
        st.write(
            "**Accuracy** — the percentage of correct predictions overall.\n\n"
            "**Precision** — of all the items the model *predicted* as a certain class, "
            "how many actually were? High precision = few false positives.\n\n"
            "**Recall** — of all the items that *actually* belong to a class, how many "
            "did the model correctly find? High recall = few false negatives.\n\n"
            "**F1 Score** — the harmonic mean of precision and recall. Balances both.\n\n"
            "**ROC-AUC** — measures how well the model distinguishes between classes "
            "across all decision thresholds. 1.0 is perfect, 0.5 is random guessing."
        )

    # ── Confusion Matrix ──────────────────────────────────────────────
    st.header("Confusion Matrix")
    st.write("Shows how many samples were correctly or incorrectly classified.")
    fig_cm = plot_confusion_matrix(y_test, y_pred_test, labels=target_names)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    with st.expander("Learn more: Reading a confusion matrix"):
        st.write(
            "Each row is the **actual** class, each column is the **predicted** class. "
            "Diagonal cells (top-left to bottom-right) are correct predictions. "
            "Off-diagonal cells are errors. For example, if row 'Cat' and column 'Dog' "
            "shows 5, it means 5 cats were incorrectly predicted as dogs."
        )

    # ── Classification Report ─────────────────────────────────────────
    st.header("Detailed Classification Report")
    report_df = classification_report_df(y_test, y_pred_test, labels=target_names)
    st.dataframe(report_df, use_container_width=True)

    # ── ROC Curve ─────────────────────────────────────────────────────
    if y_prob is not None:
        st.header("ROC Curve")
        fig_roc = plot_roc_curve(y_test, y_prob, labels=target_names)
        st.plotly_chart(fig_roc, use_container_width=True)

        with st.expander("Learn more: ROC Curve"):
            st.write(
                "The ROC curve shows the trade-off between the true positive rate "
                "(correctly identified positives) and the false positive rate (incorrectly "
                "flagged negatives) at different thresholds. A curve closer to the top-left "
                "corner is better. The dashed diagonal line represents random guessing."
            )

    # ── Feature Importance ────────────────────────────────────────────
    fig_fi = plot_feature_importance(model, feature_names)
    if fig_fi:
        st.header("Feature Importance")
        st.write("Which features had the most influence on the model's predictions.")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Learning Curve ────────────────────────────────────────────────
    st.header("Learning Curve")
    st.write("Shows how the model's performance changes with more training data.")
    with st.spinner("Computing learning curve..."):
        fig_lc = plot_learning_curve(model, X_train, y_train)
    if fig_lc:
        st.plotly_chart(fig_lc, use_container_width=True)
        with st.expander("Learn more: Reading a learning curve"):
            st.write(
                "If the **training score** is much higher than the **validation score**, "
                "the model is **overfitting** (memorizing training data). "
                "If both scores are low, the model is **underfitting** (too simple). "
                "Ideally, both curves converge at a high score."
            )
    else:
        st.info("Could not compute learning curve for this model.")


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif task_type == "regression":
    # ── Metrics ────────────────────────────────────────────────────────
    st.header("Metrics")
    m = regression_metrics(y_test, y_pred_test)

    cols = st.columns(len(m))
    for i, (name, val) in enumerate(m.items()):
        cols[i].metric(name, f"{val:.4f}")

    with st.expander("Learn more: Regression metrics explained"):
        st.write(
            "**MAE (Mean Absolute Error)** — average absolute difference between predicted "
            "and actual values. Easy to interpret: 'on average, predictions are off by X'.\n\n"
            "**MSE (Mean Squared Error)** — average of squared differences. Penalizes large "
            "errors more heavily than MAE.\n\n"
            "**RMSE (Root Mean Squared Error)** — square root of MSE. Same units as the target, "
            "easier to interpret than MSE.\n\n"
            "**R-squared** — proportion of variance explained by the model. "
            "1.0 = perfect, 0.0 = no better than predicting the mean."
        )

    # ── Actual vs Predicted ───────────────────────────────────────────
    st.header("Actual vs Predicted")
    fig_avp = plot_actual_vs_predicted(y_test, y_pred_test)
    st.plotly_chart(fig_avp, use_container_width=True)
    st.write("Points close to the red dashed line indicate accurate predictions.")

    # ── Residuals ─────────────────────────────────────────────────────
    st.header("Residual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_res = plot_residuals(y_test, y_pred_test)
        st.plotly_chart(fig_res, use_container_width=True)
    with col2:
        fig_rh = plot_residual_histogram(y_test, y_pred_test)
        st.plotly_chart(fig_rh, use_container_width=True)

    with st.expander("Learn more: Residual analysis"):
        st.write(
            "**Residuals** are the differences between actual and predicted values. "
            "Ideally, they should be randomly scattered around zero with no pattern. "
            "If you see a funnel shape or curve, the model may be missing something."
        )

    # ── Feature Importance ────────────────────────────────────────────
    fig_fi = plot_feature_importance(model, feature_names)
    if fig_fi:
        st.header("Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Learning Curve ────────────────────────────────────────────────
    st.header("Learning Curve")
    with st.spinner("Computing learning curve..."):
        fig_lc = plot_learning_curve(model, X_train, y_train)
    if fig_lc:
        st.plotly_chart(fig_lc, use_container_width=True)
    else:
        st.info("Could not compute learning curve for this model.")


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
elif task_type == "clustering":
    # ── Metrics ────────────────────────────────────────────────────────
    st.header("Clustering Quality Metrics")
    labels = y_pred_train
    m = clustering_metrics(X_train, labels)

    if "Note" in m:
        st.warning(m["Note"])
    else:
        metric_cols = st.columns(len(m))
        for i, (name, val) in enumerate(m.items()):
            if isinstance(val, float):
                metric_cols[i].metric(name, f"{val:.4f}")
            else:
                metric_cols[i].metric(name, str(val))

    with st.expander("Learn more: Clustering metrics explained"):
        st.write(
            "**Silhouette Score** (-1 to 1) — measures how similar each point is to its own "
            "cluster vs. other clusters. Higher is better. Above 0.5 is generally good.\n\n"
            "**Calinski-Harabasz Index** — ratio of between-cluster to within-cluster dispersion. "
            "Higher is better.\n\n"
            "**Davies-Bouldin Index** — average similarity between each cluster and its most similar one. "
            "Lower is better (0 means perfectly separated clusters).\n\n"
            "**Noise Points** (DBSCAN only) — points that don't belong to any cluster."
        )

    # ── Cluster Visualization ─────────────────────────────────────────
    st.header("Cluster Visualization")
    fig_clusters = plot_clusters_2d(X_train, labels, title="Training Set Clusters")
    st.plotly_chart(fig_clusters, use_container_width=True)

    if X_train.shape[1] > 2:
        st.info("Data has more than 2 features, so PCA was used to project it to 2D for visualization.")

    # ── Elbow Plot (K-Means only) ─────────────────────────────────────
    if result["model_type"] == "K-Means":
        st.header("Elbow Method")
        st.write(
            "The elbow method helps choose the right number of clusters. "
            "Look for a 'bend' or 'elbow' in the curve — that's often a good K."
        )
        with st.spinner("Computing elbow plot..."):
            fig_elbow = plot_elbow(X_train, max_k=10)
        st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Test Set Clusters ─────────────────────────────────────────────
    if y_pred_test is not None:
        st.header("Test Set Predictions")
        fig_test = plot_clusters_2d(X_test, y_pred_test, title="Test Set Cluster Assignments")
        st.plotly_chart(fig_test, use_container_width=True)


# ── Navigation ─────────────────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns(2)
with col1:
    if len(trained) >= 2:
        st.page_link("pages/08_model_comparison.py", label="Compare Models", icon=":material/compare:")
with col2:
    st.page_link("pages/09_export_summary.py", label="Export & Summary", icon=":material/download:")
