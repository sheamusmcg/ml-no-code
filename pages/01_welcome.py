import streamlit as st

st.title("Welcome to ML No Code")
st.write("*A hands-on, no-code introduction to machine learning using scikit-learn.*")

st.write(
    "This tool walks you through the complete machine learning pipeline — from loading data "
    "to training models and evaluating results — all without writing a single line of code. "
    "Every step includes plain-English explanations designed for absolute beginners."
)

# ── Pipeline Overview ──────────────────────────────────────────────────────
st.header("The ML Pipeline")
st.write("Here's the journey you'll take:")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("### 1. Load")
    st.write("Pick a dataset or upload your own CSV")
with col2:
    st.markdown("### 2. Explore")
    st.write("Understand your data with stats and charts")
with col3:
    st.markdown("### 3. Prepare")
    st.write("Clean and transform your data")
with col4:
    st.markdown("### 4. Train")
    st.write("Pick an algorithm and train a model")
with col5:
    st.markdown("### 5. Evaluate")
    st.write("See how well your model performs")

# ── What You'll Learn ─────────────────────────────────────────────────────
st.header("What You'll Learn")
st.write(
    "- What **features** and **targets** are, and how data is structured for ML\n"
    "- How to spot patterns and issues in data through **exploratory analysis**\n"
    "- Why and how to **preprocess** data (handle missing values, scale features, encode categories)\n"
    "- The difference between **classification**, **regression**, and **clustering**\n"
    "- How to train models like Decision Trees, Random Forests, KNN, SVM, and more\n"
    "- How to read **evaluation metrics** (accuracy, R-squared, silhouette score, etc.)\n"
    "- How to **compare** multiple models to find the best one"
)

# ── Glossary ──────────────────────────────────────────────────────────────
st.header("Key Terms")

with st.expander("What is Machine Learning?"):
    st.write(
        "Machine learning is a way for computers to learn patterns from data, "
        "rather than being explicitly programmed with rules. Instead of saying "
        "'if temperature > 30, then hot', you give the computer lots of examples "
        "and let it figure out the pattern on its own."
    )

with st.expander("What is a Feature?"):
    st.write(
        "A **feature** is a measurable property of something you're studying. "
        "For example, if you're predicting house prices, features might include "
        "square footage, number of bedrooms, and neighborhood. Features are the "
        "inputs to your model."
    )

with st.expander("What is a Target?"):
    st.write(
        "The **target** (also called the label) is the thing you want to predict. "
        "In a house price example, the target is the price. In a spam detector, "
        "the target is 'spam' or 'not spam'."
    )

with st.expander("What is a Model?"):
    st.write(
        "A **model** is the result of training an algorithm on data. It captures "
        "the patterns the algorithm found. Once trained, you can give the model "
        "new data and it will make predictions based on those patterns."
    )

with st.expander("What is Training vs Testing?"):
    st.write(
        "**Training** is the process of showing the model examples so it can learn patterns. "
        "**Testing** is checking how well the model works on data it has never seen before. "
        "We split data into a training set and a test set to get an honest evaluation."
    )

with st.expander("What is Overfitting?"):
    st.write(
        "**Overfitting** happens when a model memorizes the training data instead of "
        "learning general patterns. It performs great on training data but poorly on "
        "new data. It's like memorizing answers to a practice test instead of understanding "
        "the subject — you'll fail when the real test has different questions."
    )

# ── Get Started ───────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/02_data_loading.py", label="Get Started: Load Data", icon=":material/arrow_forward:")
