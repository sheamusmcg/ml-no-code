# Machine Learning Explainer 

A no-code machine learning teaching tool built with Streamlit and scikit-learn. Walk through the complete ML pipeline, from loading data to training models and making predictions, without writing a single line of code.

Designed for complete beginners. Every step includes plain-English explanations, tooltips, and guided walkthroughs.

## Live Demo

[Try it here](https://ml-explainer.streamlit.app/)

## Features

### Full ML Pipeline
- **Load Data** — 9 built-in datasets (Iris, Wine, Breast Cancer, Digits, Diabetes, California Housing, Blobs, Moons, Circles) or upload your own CSV
- **Explore Data** — summary statistics, histograms, box plots, correlation heatmaps, scatter matrices, class balance charts
- **Preprocess** — handle missing values, scale features (Standard/MinMax/Robust), encode categoricals (Label/One-Hot), remove low-variance features
- **Choose Task** — classification, regression, or clustering with automatic encoding and missing value handling
- **Train Models** — 17 algorithms with interactive hyperparameter controls:
  - Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting
  - Regression: Linear, Ridge, Lasso, Decision Tree, Random Forest, SVR, Gradient Boosting
  - Clustering: K-Means, DBSCAN, Agglomerative, Mean Shift
- **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves, residual plots, silhouette scores, learning curves, feature importance
- **Compare Models** — side-by-side metrics table, bar charts, radar charts, confusion matrices
- **Export** — download predictions, metrics, and trained models
- **Predict** — test your model on new data via manual input or CSV upload, simulating real-world deployment

### Beginner-Friendly Design
- Plain-English explanations at every step
- "Learn more" expandable sections for deeper concepts
- Tooltips on every interactive control
- Guided navigation with prerequisite checks
- Smart feature filter for high-dimensional datasets (shows top N by importance)

## Run Locally

```bash
git clone https://github.com/sheamusmcg/ml-no-code.git
cd ml-no-code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## Project Structure

```
├── streamlit_app.py           # App entrypoint and navigation
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml     # Theme configuration
├── pages/
│   ├── 01_welcome.py          # Introduction and glossary
│   ├── 02_data_loading.py     # Dataset selection and CSV upload
│   ├── 03_data_exploration.py # EDA visualizations
│   ├── 04_preprocessing.py    # Data cleaning and transformation
│   ├── 05_task_selection.py   # Task type, features, train/test split
│   ├── 06_model_training.py   # Model selection and training
│   ├── 07_evaluation.py       # Metrics and visualizations
│   ├── 08_model_comparison.py # Side-by-side model comparison
│   ├── 09_export_summary.py   # Download results
│   └── 10_predict.py          # Predict on new data
└── components/
    ├── state_manager.py       # Session state management
    ├── data_utils.py          # Dataset loaders
    ├── preprocessing_utils.py # Transformation wrappers
    ├── model_registry.py      # Model definitions and hyperparameters
    ├── evaluation_utils.py    # Metrics and plot generation
    ├── tooltips.py            # Widget help strings
    ├── explanations.py        # Concept explanations
    └── ui_helpers.py          # Reusable UI components
```

## Tech Stack

- **UI:** Streamlit
- **ML:** scikit-learn
- **Data:** pandas, numpy
- **Visualization:** Plotly, Seaborn, Matplotlib

## License

MIT
