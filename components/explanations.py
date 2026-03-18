"""Long-form markdown explanations for ML concepts.

Used inside st.expander("Learn more: ...") blocks throughout the app.
"""

TRAIN_TEST_SPLIT = """
### Why Split the Data?

Imagine studying for an exam using a practice test. If you memorize the exact answers
to the practice test, you might score perfectly on it -- but that doesn't mean you truly
understand the material. If the real exam has different questions, you'll struggle.

Machine learning models face the same challenge. If we test a model on the same data it
was trained on, it might look great but actually be just "memorizing" the training data.

By **splitting** the data into a **training set** and a **test set**, we can check if
the model has genuinely learned the underlying pattern. The test set acts as the "real exam"
-- data the model has never seen before.

A common split is **80% training / 20% testing**.
"""

OVERFITTING = """
### What is Overfitting?

**Overfitting** happens when a model is too complex for the data and starts memorizing
noise or random fluctuations instead of learning the true pattern.

Signs of overfitting:
- Very high score on training data, but much lower on test data
- The learning curve shows a large gap between training and validation scores

How to reduce overfitting:
- Use simpler models (fewer parameters, shallower trees)
- Get more training data
- Apply regularization (like Ridge or Lasso for regression)
- Remove noisy or irrelevant features
"""

FEATURE_SCALING = """
### Why Scale Features?

Features often have very different ranges. For example, "age" might range from 0 to 100,
while "annual income" might range from 0 to 1,000,000.

Some algorithms (like KNN, SVM, and neural networks) calculate distances between data points.
If income has values in the thousands while age has values below 100, the income feature will
dominate the distance calculation simply because its numbers are bigger -- not because it's
more important.

**Scaling** puts all features on a similar range so that no single feature dominates just
because of its numeric scale.

- **StandardScaler**: Centers data at mean=0, std=1 (most common)
- **MinMaxScaler**: Scales to a 0-1 range
- **RobustScaler**: Uses median and IQR, less affected by outliers
"""

CONFUSION_MATRIX = """
### Reading a Confusion Matrix

A confusion matrix shows how many predictions fell into each combination of actual vs.
predicted class.

- **Rows** = actual (true) class
- **Columns** = predicted class
- **Diagonal** (top-left to bottom-right) = correct predictions
- **Off-diagonal** = errors

For example, if the cell at row "Cat" and column "Dog" shows 5, it means 5 actual cats
were incorrectly predicted as dogs.
"""

CLUSTERING = """
### What is Clustering?

Clustering is an **unsupervised learning** technique -- it finds patterns in data
without being told what the "right answer" is.

Instead of predicting a label (like classification), clustering groups similar data points
together. You might discover that your customers naturally fall into 3 segments, or that
your documents cover 5 main topics.

Since there's no "correct" answer to compare against, we use special metrics:
- **Silhouette Score**: How similar each point is to its own cluster vs. other clusters
- **Elbow Method**: Helps choose the right number of clusters for K-Means
"""
