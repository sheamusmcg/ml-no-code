"""Reusable UI patterns for the ML No Code app."""

import streamlit as st


def require_data():
    """Show a warning and stop if no data is loaded."""
    from components.state_manager import is_data_loaded
    if not is_data_loaded():
        st.warning("Please load a dataset first.")
        st.page_link("pages/02_data_loading.py", label="Go to Load Data", icon=":material/arrow_back:")
        st.stop()


def require_model():
    """Show a warning and stop if no model is trained."""
    from components.state_manager import has_trained_models
    if not has_trained_models():
        st.warning("Please train a model first.")
        st.page_link("pages/06_model_training.py", label="Go to Train Model", icon=":material/arrow_back:")
        st.stop()


def next_step_button(page_path: str, label: str):
    """Render a navigation link to the next step."""
    st.divider()
    st.page_link(page_path, label=label, icon=":material/arrow_forward:")
