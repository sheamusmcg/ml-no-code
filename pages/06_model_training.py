import time
import streamlit as st
import numpy as np
from components.state_manager import is_data_split, is_task_configured
from components.model_registry import get_models_for_task, build_model

# ── Prerequisite check ─────────────────────────────────────────────────────
if not is_task_configured():
    st.title("Train a Model")
    st.warning("Please choose a task type first.")
    st.page_link("pages/05_task_selection.py", label="Go to Choose Task", icon=":material/arrow_back:")
    st.stop()

if not is_data_split():
    st.title("Train a Model")
    st.warning("Please prepare your data first (split into training and test sets).")
    st.page_link("pages/05_task_selection.py", label="Go to Choose Task", icon=":material/arrow_back:")
    st.stop()

task_type = st.session_state["task_type"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state.get("y_train")
y_test = st.session_state.get("y_test")

st.title("Train a Model")
st.write(
    "Now for the fun part! Pick a machine learning model, adjust its settings, "
    "and train it on your data. You can train multiple models and compare them later."
)

# ── Model Selection ───────────────────────────────────────────────────────
st.header("1. Choose a Model")

models = get_models_for_task(task_type)
model_names = list(models.keys())
selected_model = st.selectbox(
    "Model",
    model_names,
    help="Pick an algorithm to train on your data.",
)

model_info = models[selected_model]
st.info(model_info["description"])

# ── Hyperparameters ───────────────────────────────────────────────────────
st.header("2. Adjust Settings (Hyperparameters)")

with st.expander("Learn more: What are hyperparameters?"):
    st.write(
        "**Hyperparameters** are settings you choose *before* training. They control how "
        "the model learns. For example, the number of trees in a Random Forest, or how deep "
        "each tree can grow. Different settings can lead to very different results. "
        "The defaults below are reasonable starting points."
    )

user_params = {}
params_def = model_info.get("params", {})

if not params_def:
    st.write("This model has no adjustable hyperparameters.")
else:
    cols = st.columns(min(len(params_def), 3))
    for i, (param_name, pdef) in enumerate(params_def.items()):
        col = cols[i % len(cols)]
        with col:
            if pdef["type"] == "slider":
                # Determine if we should use float or int
                if isinstance(pdef["default"], float) or isinstance(pdef.get("step"), float):
                    val = st.slider(
                        param_name,
                        min_value=float(pdef["min"]),
                        max_value=float(pdef["max"]),
                        value=float(pdef["default"]),
                        step=float(pdef.get("step", 0.01)),
                        help=pdef.get("help", ""),
                        key=f"param_{selected_model}_{param_name}",
                    )
                else:
                    val = st.slider(
                        param_name,
                        min_value=int(pdef["min"]),
                        max_value=int(pdef["max"]),
                        value=int(pdef["default"]),
                        step=int(pdef.get("step", 1)),
                        help=pdef.get("help", ""),
                        key=f"param_{selected_model}_{param_name}",
                    )
                user_params[param_name] = val
            elif pdef["type"] == "select":
                val = st.selectbox(
                    param_name,
                    pdef["options"],
                    index=pdef["options"].index(pdef["default"]),
                    help=pdef.get("help", ""),
                    key=f"param_{selected_model}_{param_name}",
                )
                user_params[param_name] = val
            elif pdef["type"] == "number":
                val = st.number_input(
                    param_name,
                    value=pdef["default"],
                    help=pdef.get("help", ""),
                    key=f"param_{selected_model}_{param_name}",
                )
                user_params[param_name] = val

# ── Model Name ────────────────────────────────────────────────────────────
st.header("3. Name This Run")

# Generate a default name
existing = list(st.session_state.get("trained_models", {}).keys())
default_name = selected_model
counter = 2
while default_name in existing:
    default_name = f"{selected_model} ({counter})"
    counter += 1

run_name = st.text_input(
    "Model run name",
    value=default_name,
    help="Give this training run a name so you can identify it later when comparing models.",
)

# ── Train ─────────────────────────────────────────────────────────────────
st.header("4. Train!")

if st.button("Train Model", type="primary"):
    if run_name in st.session_state.get("trained_models", {}):
        st.warning(f"A model named '{run_name}' already exists. It will be overwritten.")

    with st.spinner(f"Training {selected_model}..."):
        try:
            model = build_model(task_type, selected_model, user_params)
            start_time = time.time()

            if task_type == "clustering":
                model.fit(X_train)
                train_labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
                # Predict on test set
                if hasattr(model, "predict"):
                    test_labels = model.predict(X_test)
                else:
                    test_labels = None
                y_pred_train = train_labels
                y_pred_test = test_labels
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

            elapsed = time.time() - start_time

            # Store results
            result = {
                "model": model,
                "model_type": selected_model,
                "task_type": task_type,
                "params": user_params,
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test,
                "train_time": elapsed,
                "feature_columns": st.session_state["feature_columns"],
            }

            # Store probability predictions for classification (if available)
            if task_type == "classification" and hasattr(model, "predict_proba"):
                result["y_prob_test"] = model.predict_proba(X_test)
                result["y_prob_train"] = model.predict_proba(X_train)

            if "trained_models" not in st.session_state:
                st.session_state["trained_models"] = {}
            st.session_state["trained_models"][run_name] = result
            st.session_state["current_model_name"] = run_name

            st.success(f"Training complete in {elapsed:.2f} seconds!")

        except Exception as e:
            st.error(f"Training failed: {e}")

# ── Trained Models Summary ────────────────────────────────────────────────
trained = st.session_state.get("trained_models", {})
if trained:
    st.divider()
    st.header("Trained Models")
    st.write(f"You have trained **{len(trained)}** model(s) so far:")

    for name, result in trained.items():
        marker = " (latest)" if name == st.session_state.get("current_model_name") else ""
        st.write(
            f"- **{name}**{marker} — {result['model_type']}, "
            f"trained in {result['train_time']:.2f}s"
        )

    st.divider()
    col_eval, col_compare = st.columns(2)
    with col_eval:
        st.page_link("pages/07_evaluation.py", label="Next: Evaluate Model", icon=":material/arrow_forward:")
    with col_compare:
        if len(trained) >= 2:
            st.page_link("pages/08_model_comparison.py", label="Compare Models", icon=":material/compare:")
