import streamlit as st
from components.state_manager import is_data_loaded, clear_downstream
from components.data_utils import BUILTIN_DATASETS, load_builtin_dataset, load_csv

st.title("Load Data")
st.write(
    "Every machine learning project starts with data. On this page, you can pick a "
    "practice dataset that comes built-in, or upload your own CSV file."
)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_builtin, tab_upload = st.tabs(["Built-in Datasets", "Upload CSV"])

# ── Built-in Datasets ──────────────────────────────────────────────────────
with tab_builtin:
    category = st.selectbox(
        "Dataset category",
        list(BUILTIN_DATASETS.keys()),
        help="Classification = predict a category, Regression = predict a number, "
             "Clustering = find groups.",
    )

    dataset_names = list(BUILTIN_DATASETS[category].keys())
    name = st.selectbox("Dataset", dataset_names)

    info = BUILTIN_DATASETS[category][name]
    st.info(info["description"])

    # Configurable params for synthetic datasets
    blob_params = {}
    if info.get("configurable"):
        col1, col2, col3 = st.columns(3)
        with col1:
            blob_params["n_samples"] = st.slider(
                "Number of samples", 100, 2000, 300, step=100,
                help="How many data points to generate.",
            )
        with col2:
            blob_params["n_clusters"] = st.slider(
                "Number of clusters", 2, 8, 3,
                help="How many distinct groups to create.",
            )
        with col3:
            blob_params["n_features"] = st.slider(
                "Number of features", 2, 10, 2,
                help="How many measurements per data point.",
            )

    if st.button("Load Dataset", type="primary", key="load_builtin"):
        with st.spinner("Loading..."):
            try:
                df = load_builtin_dataset(category, name, **blob_params)
                clear_downstream("data")
                st.session_state["data_raw"] = df
                st.session_state["data_name"] = name
                st.session_state["data_description"] = info["description"]
                st.success(
                    f"Loaded **{name}** — {df.shape[0]:,} rows, {df.shape[1]} columns."
                )
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

# ── Upload CSV ─────────────────────────────────────────────────────────────
with tab_upload:
    st.write(
        "Upload a CSV file from your computer. The first row should contain column names."
    )
    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Comma-separated values file. Maximum 50 MB.",
    )

    if uploaded is not None:
        try:
            df = load_csv(uploaded)
            st.write("**Preview** (first 10 rows):")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Use This Dataset", type="primary", key="load_csv"):
                clear_downstream("data")
                st.session_state["data_raw"] = df
                st.session_state["data_name"] = uploaded.name
                st.session_state["data_description"] = "User-uploaded CSV file."
                st.success(
                    f"Loaded **{uploaded.name}** — {df.shape[0]:,} rows, {df.shape[1]} columns."
                )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# ── Current dataset status ─────────────────────────────────────────────────
st.divider()
if is_data_loaded():
    st.write(f"**Current dataset:** {st.session_state['data_name']}")
    st.dataframe(st.session_state["data_raw"].head(5), use_container_width=True)
    st.page_link("pages/03_data_exploration.py", label="Next: Explore Data", icon=":material/arrow_forward:")
else:
    st.write("No dataset loaded yet. Pick one above to get started.")
