"""Metric computation and plot generation for model evaluation."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import learning_curve


# ── Classification Metrics ─────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification metrics."""
    result = {
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "Precision (weighted)": metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score (weighted)": metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # ROC-AUC for binary or multiclass with probabilities
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                result["ROC-AUC"] = metrics.roc_auc_score(y_true, y_prob[:, 1])
            else:
                result["ROC-AUC (OVR)"] = metrics.roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted",
                )
        except (ValueError, IndexError):
            pass
    return result


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Return a matplotlib figure of the confusion matrix."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    display_labels = labels if labels is not None else sorted(np.unique(y_true))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=display_labels, yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, labels=None):
    """Return a Plotly figure of the ROC curve."""
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
    fig = go.Figure()

    if n_classes == 2:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:, 1])
        auc = metrics.auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc:.3f}", mode="lines"))
    else:
        from sklearn.preprocessing import label_binarize
        classes = sorted(np.unique(y_true))
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = metrics.roc_curve(y_bin[:, i], y_prob[:, i])
            auc = metrics.auc(fpr, tpr)
            label = labels[i] if labels and i < len(labels) else f"Class {cls}"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{label} (AUC={auc:.3f})", mode="lines"))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def classification_report_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return the classification report as a DataFrame."""
    report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose().round(3)
    return df


# ── Regression Metrics ─────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": metrics.mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "R-squared": metrics.r2_score(y_true, y_pred),
    }


def plot_actual_vs_predicted(y_true, y_pred):
    """Scatter plot of actual vs predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(size=5, opacity=0.6), name="Predictions",
    ))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(dash="dash", color="red"), name="Perfect Prediction",
    ))
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=450,
    )
    return fig


def plot_residuals(y_true, y_pred):
    """Scatter plot of residuals vs predicted values."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(size=5, opacity=0.6), name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        height=450,
    )
    return fig


def plot_residual_histogram(y_true, y_pred):
    """Histogram of residuals."""
    residuals = y_true - y_pred
    fig = px.histogram(residuals, nbins=30, title="Residual Distribution")
    fig.update_layout(
        xaxis_title="Residual", yaxis_title="Count", height=400,
        showlegend=False,
    )
    return fig


# ── Clustering Metrics ─────────────────────────────────────────────────────

def clustering_metrics(X, labels) -> dict:
    """Compute clustering quality metrics."""
    result = {}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        result["Note"] = "Need at least 2 clusters for metrics."
        return result

    try:
        result["Silhouette Score"] = metrics.silhouette_score(X, labels)
    except ValueError:
        pass
    try:
        result["Calinski-Harabasz Index"] = metrics.calinski_harabasz_score(X, labels)
    except ValueError:
        pass
    try:
        result["Davies-Bouldin Index"] = metrics.davies_bouldin_score(X, labels)
    except ValueError:
        pass

    result["Number of Clusters"] = n_clusters
    if -1 in labels:
        result["Noise Points"] = int((np.array(labels) == -1).sum())

    return result


def plot_clusters_2d(X, labels, title="Cluster Visualization"):
    """2D scatter plot of clusters. Uses first 2 features or PCA if >2."""
    X_arr = np.array(X)
    if X_arr.shape[1] > 2:
        from sklearn.decomposition import PCA
        X_2d = PCA(n_components=2).fit_transform(X_arr)
        x_label, y_label = "PCA Component 1", "PCA Component 2"
    else:
        X_2d = X_arr[:, :2]
        x_label, y_label = "Feature 1", "Feature 2"

    df = pd.DataFrame({
        x_label: X_2d[:, 0],
        y_label: X_2d[:, 1],
        "Cluster": [str(l) for l in labels],
    })
    fig = px.scatter(
        df, x=x_label, y=y_label, color="Cluster",
        title=title, height=500,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    return fig


def plot_elbow(X, max_k=10):
    """Elbow plot for K-Means: inertia vs number of clusters."""
    from sklearn.cluster import KMeans
    inertias = []
    K_range = range(2, min(max_k + 1, len(X)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(K_range), y=inertias, mode="lines+markers", name="Inertia",
    ))
    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (within-cluster sum of squares)",
        height=400,
    )
    return fig


# ── Feature Importance / Coefficients ──────────────────────────────────────

def plot_feature_importance(model, feature_names):
    """Bar chart of feature importances or coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importances"
    elif hasattr(model, "coef_"):
        importances = model.coef_.ravel()
        if len(importances) != len(feature_names):
            # Multi-class: average absolute coefficients
            importances = np.abs(model.coef_).mean(axis=0)
        title = "Feature Coefficients (absolute for multi-class)"
    else:
        return None

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title=title, height=max(300, len(feature_names) * 25),
    )
    return fig


# ── Learning Curve ─────────────────────────────────────────────────────────

def plot_learning_curve(model, X, y, cv=5):
    """Plot training and validation scores vs training set size."""
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy" if hasattr(y, "dtype") and np.issubdtype(y.dtype, np.integer) else "r2",
            n_jobs=-1,
        )
    except Exception:
        return None

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean, mode="lines+markers",
        name="Training Score",
        line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean, mode="lines+markers",
        name="Validation Score",
        line=dict(color="orange"),
    ))
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill="toself", fillcolor="rgba(0,0,255,0.1)",
        line=dict(color="rgba(255,255,255,0)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill="toself", fillcolor="rgba(255,165,0,0.1)",
        line=dict(color="rgba(255,255,255,0)"), showlegend=False,
    ))

    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        height=450,
    )
    return fig
