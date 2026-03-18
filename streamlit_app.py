import streamlit as st
from components.state_manager import init_state

st.set_page_config(
    page_title="ML No Code",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

pages = {
    "Getting Started": [
        st.Page("pages/01_welcome.py", title="Welcome", icon=":material/school:"),
    ],
    "Data": [
        st.Page("pages/02_data_loading.py", title="Load Data", icon=":material/upload_file:"),
        st.Page("pages/03_data_exploration.py", title="Explore Data", icon=":material/query_stats:"),
        st.Page("pages/04_preprocessing.py", title="Preprocess", icon=":material/build:"),
    ],
    "Modeling": [
        st.Page("pages/05_task_selection.py", title="Choose Task", icon=":material/target:"),
        st.Page("pages/06_model_training.py", title="Train Model", icon=":material/model_training:"),
    ],
    "Results": [
        st.Page("pages/07_evaluation.py", title="Evaluate", icon=":material/assessment:"),
        st.Page("pages/08_model_comparison.py", title="Compare Models", icon=":material/compare:"),
        st.Page("pages/09_export_summary.py", title="Export & Summary", icon=":material/download:"),
    ],
}

page = st.navigation(pages)
page.run()
