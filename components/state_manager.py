import streamlit as st


def init_state():
    """Initialize all session state keys with defaults. Called once per app load."""
    defaults = {
        # Data
        "data_raw": None,
        "data_name": None,
        "data_description": None,
        "data_processed": None,
        "preprocessing_steps": [],
        # Task
        "task_type": None,
        "target_column": None,
        "feature_columns": [],
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "test_size": 0.2,
        "random_seed": 42,
        # Models
        "trained_models": {},
        "current_model_name": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_data_loaded() -> bool:
    return st.session_state.get("data_raw") is not None


def is_data_processed() -> bool:
    return st.session_state.get("data_processed") is not None


def get_working_data():
    """Return processed data if available, otherwise raw data."""
    if is_data_processed():
        return st.session_state["data_processed"]
    return st.session_state.get("data_raw")


def is_task_configured() -> bool:
    return st.session_state.get("task_type") is not None


def is_data_split() -> bool:
    return st.session_state.get("X_train") is not None


def has_trained_models() -> bool:
    return len(st.session_state.get("trained_models", {})) > 0


def clear_downstream(from_stage: str):
    """Clear session state for stages downstream of a given stage.
    Prevents stale results when the user changes an earlier step."""
    stages = {
        "data": ["data_processed", "preprocessing_steps", "task_type",
                  "target_column", "feature_columns", "X_train", "X_test",
                  "y_train", "y_test", "trained_models", "current_model_name"],
        "preprocessing": ["task_type", "target_column", "feature_columns",
                          "X_train", "X_test", "y_train", "y_test",
                          "trained_models", "current_model_name"],
        "task": ["X_train", "X_test", "y_train", "y_test",
                 "trained_models", "current_model_name"],
        "split": ["trained_models", "current_model_name"],
    }
    keys_to_clear = stages.get(from_stage, [])
    defaults = {"preprocessing_steps": [], "feature_columns": [],
                "trained_models": {}}
    for key in keys_to_clear:
        st.session_state[key] = defaults.get(key, None)
