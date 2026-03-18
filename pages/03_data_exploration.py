import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from components.state_manager import is_data_loaded
from components.data_utils import get_column_types

# ── Prerequisite check ─────────────────────────────────────────────────────
if not is_data_loaded():
    st.title("Explore Data")
    st.warning("Please load a dataset first.")
    st.page_link("pages/02_data_loading.py", label="Go to Load Data", icon=":material/arrow_back:")
    st.stop()

df = st.session_state["data_raw"]
col_types = get_column_types(df)

st.title("Explore Data")
st.write(
    "Before building a model, it is important to understand your data. This page shows "
    "summary statistics, distributions, and relationships between features."
)

# ── Dataset Overview ───────────────────────────────────────────────────────
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]}")
col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

with st.expander("Learn more: What do rows and columns mean?"):
    st.write(
        "Each **row** is one example (also called a sample or observation). "
        "Each **column** is a measurement or property of that example (also called a feature). "
        "For instance, in a flower dataset, each row is one flower and each column is a "
        "measurement like petal length."
    )

st.subheader("First 20 Rows")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Summary Statistics")
st.dataframe(df.describe(include="all").round(2), use_container_width=True)

with st.expander("Learn more: Reading summary statistics"):
    st.write(
        "**count** = number of non-missing values. **mean** = average. "
        "**std** = standard deviation (how spread out values are). "
        "**min/max** = smallest/largest value. **25%/50%/75%** = quartiles "
        "(the value below which 25%/50%/75% of data falls)."
    )

# ── Data Types ─────────────────────────────────────────────────────────────
st.subheader("Column Types")
dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.notnull().sum().values,
    "Missing": df.isnull().sum().values,
    "Unique Values": df.nunique().values,
})
st.dataframe(dtype_df, use_container_width=True, hide_index=True)

# ── Missing Values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
if missing.sum() > 0:
    st.subheader("Missing Values")
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    missing_df["% Missing"] = (missing_df["Missing Count"] / len(df) * 100).round(1)
    fig = px.bar(
        missing_df, x="Column", y="Missing Count",
        text="% Missing", title="Missing Values by Column",
        color="Missing Count", color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No missing values found in this dataset.")

# ── Distribution Plots ─────────────────────────────────────────────────────
st.header("Feature Distributions")
st.write("Select a column to see how its values are distributed.")

if col_types["numeric"]:
    selected_col = st.selectbox(
        "Choose a numeric column",
        col_types["numeric"],
        help="Pick a column to plot its histogram and box plot.",
    )

    with st.expander("Learn more: What is a distribution?"):
        st.write(
            "A **distribution** shows how often each value (or range of values) appears in your data. "
            "A **histogram** groups values into bins and counts how many fall into each bin. "
            "A **box plot** shows the median, quartiles, and any outliers at a glance."
        )

    hist_col, box_col = st.columns(2)
    with hist_col:
        fig_hist = px.histogram(
            df, x=selected_col, nbins=30,
            title=f"Histogram of {selected_col}",
            marginal="rug",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with box_col:
        fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
        st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("No numeric columns found for distribution plots.")

# ── Correlation Heatmap ────────────────────────────────────────────────────
if len(col_types["numeric"]) >= 2:
    st.header("Correlation Matrix")
    st.write("See how strongly each pair of numeric features is related.")

    with st.expander("Learn more: What is correlation?"):
        st.write(
            "**Correlation** measures how two variables move together, on a scale from -1 to +1. "
            "+1 means they increase together perfectly. -1 means one increases as the other decreases. "
            "0 means no linear relationship. High correlation between features can sometimes cause "
            "problems for certain models."
        )

    numeric_df = df[col_types["numeric"]]
    corr = numeric_df.corr()

    fig_corr, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # Scatter matrix for selected features
    st.subheader("Scatter Matrix")
    max_features = min(6, len(col_types["numeric"]))
    scatter_cols = st.multiselect(
        "Select features for scatter matrix (2-6)",
        col_types["numeric"],
        default=col_types["numeric"][:min(4, len(col_types["numeric"]))],
        help="Pick 2 to 6 numeric columns to see pairwise scatter plots.",
    )
    if len(scatter_cols) >= 2:
        fig_scatter = px.scatter_matrix(
            df, dimensions=scatter_cols[:6],
            title="Pairwise Scatter Matrix",
            height=600,
        )
        fig_scatter.update_traces(diagonal_visible=True, marker=dict(size=3))
        st.plotly_chart(fig_scatter, use_container_width=True)
    elif scatter_cols:
        st.info("Please select at least 2 columns.")

# ── Class Balance (for classification-like targets) ────────────────────────
# Check if there's a likely target column with few unique values
potential_targets = [
    c for c in df.columns
    if df[c].nunique() <= 20 and df[c].nunique() >= 2
]

if potential_targets:
    st.header("Value Counts (Potential Target Columns)")
    st.write("Columns with a small number of unique values could be targets for classification.")

    target_col = st.selectbox(
        "Select column", potential_targets,
        index=len(potential_targets) - 1,  # Default to last column (often the target)
        help="Pick a column to see the count of each unique value.",
    )
    value_counts = df[target_col].value_counts().reset_index()
    value_counts.columns = [target_col, "Count"]
    fig_bar = px.bar(
        value_counts, x=target_col, y="Count",
        title=f"Value Counts for '{target_col}'",
        color=target_col,
        text="Count",
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Imbalance warning
    counts = df[target_col].value_counts()
    if counts.max() / counts.min() > 3:
        st.warning(
            f"**Class imbalance detected:** the most common value appears "
            f"{counts.max() / counts.min():.1f}x more often than the least common. "
            "This can cause models to favor the majority class. Consider this during evaluation."
        )

# ── Navigation ─────────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/04_preprocessing.py", label="Next: Preprocess Data", icon=":material/arrow_forward:")
