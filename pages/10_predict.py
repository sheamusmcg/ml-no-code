import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from components.state_manager import has_trained_models

# ── Prerequisite check ─────────────────────────────────────────────────────
if not has_trained_models():
    st.title("Predict with Your Model")
    st.warning("Please train a model first.")
    st.page_link("pages/06_model_training.py", label="Go to Train Model", icon=":material/arrow_back:")
    st.stop()

trained = st.session_state["trained_models"]
task_type = list(trained.values())[0]["task_type"]

st.title("Predict with Your Model")
st.write(
    "This is how machine learning works in the real world. Once a model is trained, "
    "you can give it **new, unseen data** and it will make predictions. This page simulates "
    "deploying your model — just like a real application would use it."
)

with st.expander("Learn more: How are ML models used in production?"):
    st.write(
        "In a real application, the trained model is saved and loaded into a server or app. "
        "When new data arrives (a new customer, a new house listing, a new medical scan), "
        "the app sends the features to the model and gets back a prediction.\n\n"
        "For example:\n"
        "- A bank receives a new loan application → the model predicts risk of default\n"
        "- A real estate website gets a new listing → the model predicts the price\n"
        "- A hospital receives new lab results → the model predicts the diagnosis\n\n"
        "The key requirement is that **new data must have the same features** the model "
        "was trained on, in the same format."
    )

# ── Model Selector ────────────────────────────────────────────────────────
st.header("1. Select Your Model")
model_names = list(trained.keys())
selected = st.selectbox(
    "Which trained model should make predictions?",
    model_names,
    index=model_names.index(st.session_state.get("current_model_name", model_names[0]))
    if st.session_state.get("current_model_name") in model_names else 0,
)
result = trained[selected]
model = result["model"]
feature_names = result["feature_columns"]
target_names = st.session_state.get("target_names")

st.info(f"**Model:** {result['model_type']} | **Features:** {len(feature_names)} | **Task:** {task_type.title()}")

# ── Compute feature stats from training data for slider ranges ────────────
X_train = st.session_state["X_train"]
feature_stats = {}
for i, fname in enumerate(feature_names):
    col_data = X_train[:, i]
    feature_stats[fname] = {
        "min": float(np.nanmin(col_data)),
        "max": float(np.nanmax(col_data)),
        "mean": float(np.nanmean(col_data)),
        "median": float(np.nanmedian(col_data)),
        "is_integer": np.all(col_data == col_data.astype(int)),
    }

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_manual, tab_batch = st.tabs(["Manual Input (Single Prediction)", "Upload New Data (Batch Predictions)"])

# ═══════════════════════════════════════════════════════════════════════════
# MANUAL INPUT
# ═══════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.header("2. Enter Feature Values")
    st.write(
        "Adjust the values below to describe a new example. The model will predict "
        "the outcome based on these inputs — just like a real application would."
    )

    if len(feature_names) > 20:
        st.info(
            f"This model has {len(feature_names)} features. Showing them in a compact layout. "
            "Adjust the ones you're interested in — the rest use default (median) values."
        )

    # Build input form
    input_values = {}
    n_cols = 3 if len(feature_names) > 6 else 2 if len(feature_names) > 3 else 1
    cols = st.columns(n_cols)

    for i, fname in enumerate(feature_names):
        stats = feature_stats[fname]
        col = cols[i % n_cols]
        with col:
            range_width = stats["max"] - stats["min"]
            if range_width == 0:
                range_width = 1.0

            if stats["is_integer"] and range_width <= 100:
                # Integer-like feature: use slider
                input_values[fname] = st.slider(
                    fname,
                    min_value=int(stats["min"]),
                    max_value=max(int(stats["max"]), int(stats["min"]) + 1),
                    value=int(stats["median"]),
                    key=f"manual_{fname}",
                )
            elif stats["is_integer"] and stats["min"] >= 0 and stats["max"] <= 2:
                # Binary-like: 0/1 toggle
                input_values[fname] = st.selectbox(
                    fname,
                    [0, 1],
                    index=int(round(stats["median"])),
                    key=f"manual_{fname}",
                )
            else:
                # Continuous feature: number input
                step = round(range_width / 100, 4) if range_width > 0 else 0.01
                input_values[fname] = st.number_input(
                    fname,
                    min_value=float(stats["min"] - range_width * 0.1),
                    max_value=float(stats["max"] + range_width * 0.1),
                    value=float(round(stats["median"], 4)),
                    step=float(step),
                    key=f"manual_{fname}",
                )

    # Predict button
    st.divider()
    if st.button("Predict", type="primary", key="predict_single"):
        input_array = np.array([[input_values[f] for f in feature_names]])

        if task_type == "classification":
            prediction = model.predict(input_array)[0]
            label = target_names[prediction] if target_names and prediction < len(target_names) else str(prediction)

            st.success(f"### Prediction: **{label}**")

            # Show probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_array)[0]
                st.write("**Confidence for each class:**")
                prob_df = pd.DataFrame({
                    "Class": target_names if target_names else [str(i) for i in range(len(probs))],
                    "Probability": probs,
                    "Confidence": [f"{p:.1%}" for p in probs],
                })
                # Highlight the predicted class
                st.dataframe(
                    prob_df.sort_values("Probability", ascending=False),
                    use_container_width=True, hide_index=True,
                )

                # Bar chart
                import plotly.express as px
                fig = px.bar(
                    prob_df, x="Class", y="Probability",
                    color="Probability",
                    color_continuous_scale="Blues",
                    title="Prediction Confidence",
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        elif task_type == "regression":
            prediction = model.predict(input_array)[0]
            st.success(f"### Predicted Value: **{prediction:.4f}**")

            # Context from training data
            y_train = st.session_state.get("y_train")
            if y_train is not None:
                st.write(
                    f"For context, training data target values ranged from "
                    f"**{y_train.min():.2f}** to **{y_train.max():.2f}** "
                    f"(mean: {y_train.mean():.2f})."
                )

        elif task_type == "clustering":
            prediction = model.predict(input_array)[0] if hasattr(model, "predict") else "N/A"
            st.success(f"### Assigned to Cluster: **{prediction}**")

        with st.expander("What just happened?"):
            st.write(
                "The model took your input values, processed them through the algorithm "
                "it learned during training, and produced a prediction. In a real application, "
                "this same process would happen automatically — perhaps triggered by a form "
                "submission, an API call, or a database event."
            )


# ═══════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("2. Upload New (Unseen) Data")
    st.write(
        "Upload a CSV file with new data that the model has **never seen before**. "
        "The model will predict an outcome for every row in your file. "
        "This simulates running your model in a real production pipeline."
    )

    with st.expander("What format should my CSV be in?"):
        st.write(
            f"Your CSV should have columns that match the features the model was trained on. "
            f"The model expects **{len(feature_names)} feature(s)**.\n\n"
            "If your CSV has the original (non-encoded) columns, the app will try to "
            "encode them automatically using the same method as during training.\n\n"
            "**Tip:** You can download a sample template below to see the expected format."
        )

        # Template download
        template_df = pd.DataFrame(
            {fname: [feature_stats[fname]["median"]] for fname in feature_names}
        )
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            "Download Template CSV",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader(
        "Upload CSV for predictions",
        type=["csv"],
        help="A CSV file with the same features the model was trained on.",
        key="batch_upload",
    )

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.write(f"**Uploaded:** {new_df.shape[0]:,} rows, {new_df.shape[1]} columns")
            st.dataframe(new_df.head(10), use_container_width=True)

            # Check if columns match exactly
            missing_cols = [c for c in feature_names if c not in new_df.columns]
            extra_cols = [c for c in new_df.columns if c not in feature_names]

            work_df = new_df.copy()
            needs_encoding = False

            if missing_cols:
                # Maybe the user uploaded original (pre-encoded) data
                st.warning(
                    f"Your CSV is missing {len(missing_cols)} expected column(s). "
                    "This may be because the training data was encoded. "
                    "The app will try to auto-encode your data."
                )
                needs_encoding = True

            # Auto-encode if needed
            if needs_encoding:
                non_numeric_cols = work_df.select_dtypes(exclude=[np.number]).columns.tolist()
                if non_numeric_cols:
                    encode_method = st.radio(
                        "How should we encode the categorical columns?",
                        ["label", "onehot", "drop"],
                        format_func=lambda x: {
                            "label": "Label Encoding",
                            "onehot": "One-Hot Encoding",
                            "drop": "Drop categorical columns",
                        }[x],
                        key="batch_encode",
                    )

                    if encode_method == "label":
                        for col in non_numeric_cols:
                            le_temp = LabelEncoder()
                            work_df[col] = le_temp.fit_transform(work_df[col].astype(str))
                    elif encode_method == "onehot":
                        work_df = pd.get_dummies(work_df, columns=non_numeric_cols, drop_first=True, dtype=int)
                    elif encode_method == "drop":
                        work_df = work_df.drop(columns=non_numeric_cols)

                # Handle missing values
                if work_df.isnull().sum().sum() > 0:
                    for col in work_df.columns:
                        if work_df[col].isnull().sum() > 0:
                            if work_df[col].dtype in [np.float64, np.int64, float, int]:
                                work_df[col] = work_df[col].fillna(work_df[col].median())
                            else:
                                work_df[col] = work_df[col].fillna(work_df[col].mode().iloc[0])

                # Try to align columns with what the model expects
                available = [c for c in feature_names if c in work_df.columns]
                missing_after = [c for c in feature_names if c not in work_df.columns]

                if missing_after:
                    st.warning(
                        f"After encoding, {len(missing_after)} feature(s) are still missing. "
                        f"These will be filled with 0: {', '.join(missing_after[:10])}"
                        + ("..." if len(missing_after) > 10 else "")
                    )
                    for col in missing_after:
                        work_df[col] = 0

                # Reorder to match training feature order
                work_df = work_df[feature_names]
            else:
                # Handle missing values in matched columns
                if work_df[feature_names].isnull().sum().sum() > 0:
                    st.info("Filling missing values with column medians.")
                    for col in feature_names:
                        if work_df[col].isnull().sum() > 0:
                            work_df[col] = work_df[col].fillna(work_df[col].median())
                work_df = work_df[feature_names]

            # ── Run predictions ───────────────────────────────────────────
            if st.button("Run Batch Predictions", type="primary", key="predict_batch"):
                X_new = work_df.values

                if task_type == "classification":
                    predictions = model.predict(X_new)
                    if target_names:
                        pred_labels = [target_names[p] if p < len(target_names) else str(p) for p in predictions]
                    else:
                        pred_labels = predictions.tolist()

                    result_df = new_df.copy()
                    result_df["Predicted Class"] = pred_labels

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_new)
                        confidence = probs.max(axis=1)
                        result_df["Confidence"] = [f"{c:.1%}" for c in confidence]

                elif task_type == "regression":
                    predictions = model.predict(X_new)
                    result_df = new_df.copy()
                    result_df["Predicted Value"] = np.round(predictions, 4)

                elif task_type == "clustering":
                    if hasattr(model, "predict"):
                        predictions = model.predict(X_new)
                    else:
                        predictions = ["N/A"] * len(X_new)
                    result_df = new_df.copy()
                    result_df["Assigned Cluster"] = predictions

                # Display results
                st.success(f"Predictions complete for **{len(result_df):,}** rows!")
                st.dataframe(result_df, use_container_width=True)

                # Summary stats
                if task_type == "classification":
                    st.subheader("Prediction Summary")
                    value_counts = pd.Series(pred_labels).value_counts()
                    import plotly.express as px
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title="Distribution of Predicted Classes",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif task_type == "regression":
                    st.subheader("Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Min", f"{predictions.min():.2f}")
                    col2.metric("Max", f"{predictions.max():.2f}")
                    col3.metric("Mean", f"{predictions.mean():.2f}")
                    col4.metric("Median", f"{np.median(predictions):.2f}")

                # Download results
                csv_result = result_df.to_csv(index=False)
                st.download_button(
                    "Download Predictions CSV",
                    data=csv_result,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    key="dl_batch_predictions",
                )

                with st.expander("What just happened?"):
                    st.write(
                        "The model processed each row of your uploaded data — applying the "
                        "same algorithm it learned during training — and produced a prediction "
                        "for every row. This is exactly how ML models work in production:\n\n"
                        "1. New data arrives\n"
                        "2. It's formatted to match the training features\n"
                        "3. The model makes predictions\n"
                        "4. Results are returned to the user or stored in a database\n\n"
                        "In a real system, this would happen via an API endpoint, a scheduled "
                        "batch job, or a streaming pipeline."
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
