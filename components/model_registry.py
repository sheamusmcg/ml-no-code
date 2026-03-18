"""Central registry of all ML models, their hyperparameters, and descriptions.

Each model entry contains:
- class_: The scikit-learn class to instantiate.
- description: A beginner-friendly explanation.
- params: Dict of hyperparameter definitions, each with:
    - type: "slider", "select", or "number"
    - For slider: min, max, default, step (optional)
    - For select: options, default
    - help: Short tooltip string
"""

from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift


MODEL_REGISTRY = {
    "classification": {
        "Logistic Regression": {
            "class_": LogisticRegression,
            "description": (
                "A fast, simple model that draws a straight boundary between classes. "
                "Good as a first baseline. Works best when classes are roughly linearly separable."
            ),
            "params": {
                "C": {
                    "type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                    "help": "Regularization strength. Smaller values = stronger regularization (simpler model).",
                },
                "max_iter": {
                    "type": "slider", "min": 100, "max": 2000, "default": 200, "step": 100,
                    "help": "Maximum number of iterations for the solver to converge.",
                },
            },
            "extra_args": {"random_state": 42},
        },
        "Decision Tree": {
            "class_": DecisionTreeClassifier,
            "description": (
                "Makes decisions by asking a series of yes/no questions about the features, "
                "like a flowchart. Easy to understand and visualize, but can overfit if too deep."
            ),
            "params": {
                "max_depth": {
                    "type": "slider", "min": 1, "max": 30, "default": 5, "step": 1,
                    "help": "Maximum depth of the tree. Deeper trees are more complex and may overfit.",
                },
                "min_samples_split": {
                    "type": "slider", "min": 2, "max": 50, "default": 2, "step": 1,
                    "help": "Minimum samples required to split a node. Higher values prevent overfitting.",
                },
                "min_samples_leaf": {
                    "type": "slider", "min": 1, "max": 50, "default": 1, "step": 1,
                    "help": "Minimum samples required in a leaf node. Higher values make simpler trees.",
                },
            },
            "extra_args": {"random_state": 42},
        },
        "Random Forest": {
            "class_": RandomForestClassifier,
            "description": (
                "Builds many decision trees and combines their votes. More robust than a single tree "
                "and less prone to overfitting. One of the most popular algorithms."
            ),
            "params": {
                "n_estimators": {
                    "type": "slider", "min": 10, "max": 500, "default": 100, "step": 10,
                    "help": "Number of trees in the forest. More trees = better but slower.",
                },
                "max_depth": {
                    "type": "slider", "min": 1, "max": 30, "default": 10, "step": 1,
                    "help": "Maximum depth of each tree.",
                },
                "min_samples_split": {
                    "type": "slider", "min": 2, "max": 20, "default": 2, "step": 1,
                    "help": "Minimum samples to split a node.",
                },
            },
            "extra_args": {"random_state": 42, "n_jobs": -1},
        },
        "K-Nearest Neighbors": {
            "class_": KNeighborsClassifier,
            "description": (
                "Classifies a sample by looking at the K closest training examples and taking a vote. "
                "Simple and intuitive, but can be slow on large datasets."
            ),
            "params": {
                "n_neighbors": {
                    "type": "slider", "min": 1, "max": 50, "default": 5, "step": 1,
                    "help": "Number of neighbors to consider. Small K = complex boundary, Large K = smoother.",
                },
                "weights": {
                    "type": "select", "options": ["uniform", "distance"], "default": "uniform",
                    "help": "'uniform' = all neighbors count equally. 'distance' = closer neighbors count more.",
                },
            },
            "extra_args": {},
        },
        "Support Vector Machine": {
            "class_": SVC,
            "description": (
                "Finds the best possible boundary (hyperplane) between classes with the widest margin. "
                "Powerful for high-dimensional data. Can use kernel tricks for non-linear boundaries."
            ),
            "params": {
                "C": {
                    "type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                    "help": "Regularization parameter. Larger C = less regularization (tighter fit).",
                },
                "kernel": {
                    "type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf",
                    "help": "'linear' = straight boundary. 'rbf' = flexible curved boundary. 'poly' = polynomial.",
                },
            },
            "extra_args": {"random_state": 42, "probability": True},
        },
        "Gradient Boosting": {
            "class_": GradientBoostingClassifier,
            "description": (
                "Builds trees one at a time, where each new tree corrects the errors of the previous ones. "
                "Often achieves high accuracy but can be slow to train."
            ),
            "params": {
                "n_estimators": {
                    "type": "slider", "min": 10, "max": 500, "default": 100, "step": 10,
                    "help": "Number of boosting rounds. More rounds = better but risk of overfitting.",
                },
                "learning_rate": {
                    "type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01,
                    "help": "How much each tree contributes. Smaller = slower learning but often better.",
                },
                "max_depth": {
                    "type": "slider", "min": 1, "max": 10, "default": 3, "step": 1,
                    "help": "Depth of each tree. Shallow trees (3-5) work well for boosting.",
                },
            },
            "extra_args": {"random_state": 42},
        },
    },
    "regression": {
        "Linear Regression": {
            "class_": LinearRegression,
            "description": (
                "Fits a straight line (or plane) through the data. The simplest regression model. "
                "Good as a baseline to see if more complex models are justified."
            ),
            "params": {},
            "extra_args": {},
        },
        "Ridge Regression": {
            "class_": Ridge,
            "description": (
                "Linear regression with L2 regularization. Prevents overfitting by penalizing "
                "large coefficients. Good when features are correlated."
            ),
            "params": {
                "alpha": {
                    "type": "slider", "min": 0.01, "max": 100.0, "default": 1.0, "step": 0.01,
                    "help": "Regularization strength. Larger alpha = simpler model.",
                },
            },
            "extra_args": {"random_state": 42},
        },
        "Lasso Regression": {
            "class_": Lasso,
            "description": (
                "Linear regression with L1 regularization. Can set some feature weights to exactly zero, "
                "effectively performing feature selection."
            ),
            "params": {
                "alpha": {
                    "type": "slider", "min": 0.01, "max": 100.0, "default": 1.0, "step": 0.01,
                    "help": "Regularization strength. Larger alpha = fewer features used.",
                },
            },
            "extra_args": {"random_state": 42},
        },
        "Decision Tree Regressor": {
            "class_": DecisionTreeRegressor,
            "description": (
                "A decision tree that predicts numbers instead of categories. Splits the data into "
                "regions and predicts the average value in each region."
            ),
            "params": {
                "max_depth": {
                    "type": "slider", "min": 1, "max": 30, "default": 5, "step": 1,
                    "help": "Maximum depth of the tree.",
                },
                "min_samples_split": {
                    "type": "slider", "min": 2, "max": 50, "default": 2, "step": 1,
                    "help": "Minimum samples required to split a node.",
                },
            },
            "extra_args": {"random_state": 42},
        },
        "Random Forest Regressor": {
            "class_": RandomForestRegressor,
            "description": (
                "Averages predictions from many decision trees. More stable and accurate "
                "than a single tree."
            ),
            "params": {
                "n_estimators": {
                    "type": "slider", "min": 10, "max": 500, "default": 100, "step": 10,
                    "help": "Number of trees.",
                },
                "max_depth": {
                    "type": "slider", "min": 1, "max": 30, "default": 10, "step": 1,
                    "help": "Maximum depth of each tree.",
                },
            },
            "extra_args": {"random_state": 42, "n_jobs": -1},
        },
        "SVR": {
            "class_": SVR,
            "description": (
                "Support Vector Regression. Finds a function that deviates from actual values by "
                "at most epsilon, while being as flat as possible."
            ),
            "params": {
                "C": {
                    "type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                    "help": "Regularization parameter.",
                },
                "kernel": {
                    "type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf",
                    "help": "Kernel type for the boundary function.",
                },
                "epsilon": {
                    "type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01,
                    "help": "Tolerance margin. Points within epsilon of the prediction are not penalized.",
                },
            },
            "extra_args": {},
        },
        "Gradient Boosting Regressor": {
            "class_": GradientBoostingRegressor,
            "description": (
                "Builds trees sequentially, each correcting errors from the previous ones. "
                "Often achieves strong performance on tabular data."
            ),
            "params": {
                "n_estimators": {
                    "type": "slider", "min": 10, "max": 500, "default": 100, "step": 10,
                    "help": "Number of boosting rounds.",
                },
                "learning_rate": {
                    "type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01,
                    "help": "Step size for each tree's contribution.",
                },
                "max_depth": {
                    "type": "slider", "min": 1, "max": 10, "default": 3, "step": 1,
                    "help": "Depth of each tree.",
                },
            },
            "extra_args": {"random_state": 42},
        },
    },
    "clustering": {
        "K-Means": {
            "class_": KMeans,
            "description": (
                "Divides data into K groups by finding cluster centers that minimize distances. "
                "Fast and simple, but you must choose K in advance. Works best with spherical clusters."
            ),
            "params": {
                "n_clusters": {
                    "type": "slider", "min": 2, "max": 15, "default": 3, "step": 1,
                    "help": "Number of clusters to find.",
                },
                "init": {
                    "type": "select", "options": ["k-means++", "random"], "default": "k-means++",
                    "help": "'k-means++' = smart initialization (recommended). 'random' = random start.",
                },
                "n_init": {
                    "type": "slider", "min": 1, "max": 20, "default": 10, "step": 1,
                    "help": "Number of times to run with different initial centers. Best result is kept.",
                },
            },
            "extra_args": {"random_state": 42},
        },
        "DBSCAN": {
            "class_": DBSCAN,
            "description": (
                "Finds clusters based on density — groups of points that are closely packed together. "
                "Can find clusters of any shape and automatically detects outliers. Does not require "
                "specifying the number of clusters."
            ),
            "params": {
                "eps": {
                    "type": "slider", "min": 0.1, "max": 5.0, "default": 0.5, "step": 0.1,
                    "help": "Maximum distance between two points to be in the same neighborhood.",
                },
                "min_samples": {
                    "type": "slider", "min": 2, "max": 20, "default": 5, "step": 1,
                    "help": "Minimum points needed to form a dense region (cluster core).",
                },
            },
            "extra_args": {},
        },
        "Agglomerative Clustering": {
            "class_": AgglomerativeClustering,
            "description": (
                "Starts with each point as its own cluster, then repeatedly merges the closest "
                "clusters until the desired number is reached. Creates a hierarchy of clusters."
            ),
            "params": {
                "n_clusters": {
                    "type": "slider", "min": 2, "max": 15, "default": 3, "step": 1,
                    "help": "Number of clusters to find.",
                },
                "linkage": {
                    "type": "select", "options": ["ward", "complete", "average", "single"],
                    "default": "ward",
                    "help": "How to measure distance between clusters. 'ward' minimizes variance (most common).",
                },
            },
            "extra_args": {},
        },
        "Mean Shift": {
            "class_": MeanShift,
            "description": (
                "Finds clusters by shifting points toward the densest area nearby, like rolling "
                "a ball uphill. Automatically determines the number of clusters."
            ),
            "params": {},
            "extra_args": {},
        },
    },
}


def get_models_for_task(task_type: str) -> dict:
    """Return the model registry entries for a given task type."""
    return MODEL_REGISTRY.get(task_type, {})


def build_model(task_type: str, model_name: str, user_params: dict):
    """Instantiate a scikit-learn model with user-selected parameters."""
    entry = MODEL_REGISTRY[task_type][model_name]
    all_params = {**entry.get("extra_args", {}), **user_params}
    return entry["class_"](**all_params)
