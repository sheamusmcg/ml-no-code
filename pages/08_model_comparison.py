import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from components.state_manager import has_trained_models
from components.evaluation_utils import (
    classification_metrics, regression_metrics, clustering_metrics,
    plot_confusion_matrix,
)

# ── Prerequisite check ─────────────────────────────────────────────────────
if not has_trained_models():
    st.title("Compare Models")
    st.warning("Please train at least one model first.")
    st.page_link("pages/06_model_training.py", label="Go to Train Model", icon=":material/arrow_back:")
    st.stop()

trained = st.session_state["trained_models"]

if len(trained) < 2:
    st.title("Compare Models")
    st.info(
        f"You have {len(trained)} model(s) trained. Train at least 2 models to compare them. "
        "Go back to the Training page and try a different algorithm or different settings."
    )
    st.page_link("pages/06_model_training.py", label="Go to Train Model", icon=":material/arrow_back:")
    st.stop()

st.title("Compare Models")
st.write(
    "See how your trained models stack up against each other. This helps you decide "
    "which model works best for your data."
)

X_test = st.session_state["X_test"]
y_test = st.session_state.get("y_test")

# ── Model Selector ────────────────────────────────────────────────────────
model_names = list(trained.keys())
selected_models = st.multiselect(
    "Models to compare",
    model_names,
    default=model_names,
    help="Select which models to include in the comparison.",
)

if len(selected_models) < 2:
    st.info("Select at least 2 models to compare.")
    st.stop()

task_type = trained[selected_models[0]]["task_type"]

# ── Summary Table ─────────────────────────────────────────────────────────
st.header("Summary Table")

rows = []
for name in selected_models:
    result = trained[name]
    row = {
        "Model": name,
        "Algorithm": result["model_type"],
        "Training Time (s)": round(result["train_time"], 3),
    }

    if task_type == "classification":
        m = classification_metrics(y_test, result["y_pred_test"], result.get("y_prob_test"))
        row.update({k: round(v, 4) for k, v in m.items()})
    elif task_type == "regression":
        m = regression_metrics(y_test, result["y_pred_test"])
        row.update({k: round(v, 4) for k, v in m.items()})
    elif task_type == "clustering":
        X_train = st.session_state["X_train"]
        m = clustering_metrics(X_train, result["y_pred_train"])
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    rows.append(row)

summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Primary Metric Bar Chart ──────────────────────────────────────────────
st.header("Primary Metric Comparison")

if task_type == "classification":
    primary_metric = "Accuracy"
elif task_type == "regression":
    primary_metric = "R-squared"
elif task_type == "clustering":
    primary_metric = "Silhouette Score"
else:
    primary_metric = None

if primary_metric and primary_metric in summary_df.columns:
    fig_bar = px.bar(
        summary_df, x="Model", y=primary_metric,
        color="Model", title=f"{primary_metric} by Model",
        text=primary_metric,
    )
    fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_bar.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Highlight best model
    if task_type == "clustering" and "Davies-Bouldin Index" in summary_df.columns:
        best_idx = summary_df[primary_metric].idxmax()
    else:
        best_idx = summary_df[primary_metric].idxmax()
    best_model = summary_df.loc[best_idx, "Model"]
    best_score = summary_df.loc[best_idx, primary_metric]
    st.success(f"Best model by {primary_metric}: **{best_model}** ({best_score:.4f})")

# ── Radar Chart (Classification only) ─────────────────────────────────────
if task_type == "classification":
    st.header("Multi-Metric Radar Chart")

    radar_metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 Score (weighted)"]
    available_radar = [m for m in radar_metrics if m in summary_df.columns]

    if len(available_radar) >= 3:
        fig_radar = go.Figure()
        for _, row in summary_df.iterrows():
            values = [row[m] for m in available_radar]
            values.append(values[0])  # Close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=available_radar + [available_radar[0]],
                fill="toself",
                name=row["Model"],
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Multi-Metric Comparison",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Side-by-Side Confusion Matrices (Classification) ──────────────────────
if task_type == "classification" and len(selected_models) <= 4:
    st.header("Confusion Matrices")
    target_names = st.session_state.get("target_names")
    cols = st.columns(min(len(selected_models), 2))
    for i, name in enumerate(selected_models):
        with cols[i % len(cols)]:
            st.subheader(name)
            fig_cm = plot_confusion_matrix(y_test, trained[name]["y_pred_test"], labels=target_names)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

# ── Side-by-Side Actual vs Predicted (Regression) ─────────────────────────
if task_type == "regression" and len(selected_models) <= 4:
    st.header("Actual vs Predicted")
    cols = st.columns(min(len(selected_models), 2))
    for i, name in enumerate(selected_models):
        with cols[i % len(cols)]:
            st.subheader(name)
            y_pred = trained[name]["y_pred_test"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(size=5, opacity=0.6),
            ))
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", line=dict(dash="dash", color="red"),
            ))
            fig.update_layout(
                xaxis_title="Actual", yaxis_title="Predicted",
                height=350, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Training Time Comparison ──────────────────────────────────────────────
st.header("Training Time")
fig_time = px.bar(
    summary_df, x="Model", y="Training Time (s)",
    color="Model", text="Training Time (s)",
    title="Training Time by Model",
)
fig_time.update_traces(texttemplate="%{text:.3f}s", textposition="outside")
fig_time.update_layout(height=400, showlegend=False)
st.plotly_chart(fig_time, use_container_width=True)

# ── Navigation ─────────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/09_export_summary.py", label="Next: Export & Summary", icon=":material/download:")
