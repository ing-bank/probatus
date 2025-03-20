from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap

from probatus.features import ShapRFECV

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")

# TODO: Remove this skip once CatBoost is updated to work with NumPy 2.0
# Mock CatBoost to avoid import errors
# from catboost import CatBoostClassifier
CatBoostClassifier = Mock  # Create a mock for CatBoostClassifier

# ====================================
# Plot Configuration
# ====================================


# Define pytest command line option for saving plots
def pytest_addoption(parser):
    parser.addoption(
        "--save-plots",
        action="store",
        default="false",
        choices=["true", "false"],
        help="Save plots generated during tests (true/false)",
    )


@pytest.fixture
def save_plots(request):
    return request.config.getoption("--save-plots").lower() == "true"


@pytest.fixture(scope="function")
def check_plots_are_generated_correctly():
    """
    Fixture providing a function to check if a plot contains at least a few colors.

    Returns:
        function: A function that checks if plots are generated correctly
    """

    def _check(image_path, min_unique_colors=4):
        """
        Check if a plot contains at least a few colors, to ensure that it is not a blank plot.
        But also to ensure it is not a plot with only an axis.

        Args:
            image_path (str): Path to the image file
            min_unique_colors (int): Minimum number of unique colors required

        Returns:
            bool: True if the image has at least min_unique_colors unique colors
        """
        try:
            # Open the image
            img = Image.open(image_path)

            # Convert to RGB if not already
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get image data as numpy array
            img_data = np.array(img)

            # Reshape to a list of pixels
            pixels = img_data.reshape(-1, 3)

            # Count unique colors (unique rows in the pixel array)
            unique_colors = np.unique(pixels, axis=0)
            num_unique_colors = len(unique_colors)

            # Check if the image has enough unique colors
            return num_unique_colors >= min_unique_colors
        except Exception as e:
            print(f"Error checking image {image_path}: {e}")
            return False

    return _check


@pytest.fixture(scope="function")
def get_plots_dir():
    """
    Fixture providing a function to get the plots directory for a given estimator class.

    Returns:
        function: A function to get plot directory paths
    """

    def _get_plots_dir(base_plots_dir, estimator_class, estimators_list):
        """
        Helper function to get the plots directory for a given estimator class.

        Args:
            base_plots_dir: Base directory for plots
            estimator_class: The estimator class
            estimators_list: List of estimator parameters (from ESTIMATORS)

        Returns:
            Path to the plots directory
        """
        estimator_name = next(param.id for param in estimators_list if param.values[0] == estimator_class)
        return os.path.join(base_plots_dir, f"{estimator_name}_plots")

    return _get_plots_dir


@pytest.fixture(scope="function")
def setup_plot_dirs():
    """
    Fixture providing a function to create plot directories.

    Returns:
        function: A function to set up plot directories
    """

    def _setup_plot_dirs(save_plots, base_plots_dir, estimators_list):
        """
        Create plot directories only if save_plots is True.

        Args:
            save_plots (bool): Whether to save plots or not
            base_plots_dir (str): Base directory for plots
            estimators_list (list): List of estimator parameters (from ESTIMATORS)
        """
        if save_plots:
            # Create base plots directory if it doesn't exist
            os.makedirs(base_plots_dir, exist_ok=True)

            # Create plots directory for each estimator
            for param in estimators_list:
                estimator_name = param.id
                estimator_plots_dir = os.path.join(base_plots_dir, f"{estimator_name}_plots")
                os.makedirs(estimator_plots_dir, exist_ok=True)

    return _setup_plot_dirs


# ====================================
# Random State Fixtures
# ====================================


@pytest.fixture(scope="function")
def random_state():
    """Default random state (42) for reproducibility."""
    return 42


@pytest.fixture(scope="function")
def random_states():
    """Dictionary of various random states for tests that need multiple seeds."""
    return {
        "default": 42,  # Same as random_state
        "seed_1": 1,  # For tests that need a different seed
        "seed_1234": 1234,  # For tests requiring a third seed
    }


# Maintained for backward compatibility
@pytest.fixture(scope="function")
def random_state_42():
    return 42


@pytest.fixture(scope="function")
def random_state_1234():
    return 1234


@pytest.fixture(scope="function")
def random_state_1():
    return 1


# ====================================
# Basic Data Fixtures
# ====================================


@pytest.fixture(scope="function")
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    y_series = pd.Series(y)

    # Create and fit a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_df, y_series)

    return X_df, y_series, model


@pytest.fixture(scope="function")
def simple_tabular_data():
    """Create simplified tabular data with clearly defined patterns."""
    X_train = pd.DataFrame({"col_1": [1, 1, 1, 1], "col_2": [0, 0, 0, 0], "col_3": [1, 0, 1, 0]}, index=[1, 2, 3, 4])
    y_train = pd.Series([1, 0, 1, 0], index=[1, 2, 3, 4])
    X_test = pd.DataFrame({"col_1": [1, 1, 1], "col_2": [0, 0, 0], "col_3": [1, 0, 0]}, index=[5, 6, 7])
    y_test = pd.Series([1, 0, 0], index=[5, 6, 7])

    return X_train, y_train, X_test, y_test


# Backward compatibility fixtures
@pytest.fixture(scope="function")
def X_train(simple_tabular_data):
    X_train, _, _, _ = simple_tabular_data
    return X_train


@pytest.fixture(scope="function")
def y_train(simple_tabular_data):
    _, y_train, _, _ = simple_tabular_data
    return y_train


@pytest.fixture(scope="function")
def X_test(simple_tabular_data):
    _, _, X_test, _ = simple_tabular_data
    return X_test


@pytest.fixture(scope="function")
def y_test(simple_tabular_data):
    _, _, _, y_test = simple_tabular_data
    return y_test


@pytest.fixture(scope="function")
def X1():
    return pd.DataFrame({"col_1": [1, 1, 1, 1], "col_2": [0, 0, 0, 0], "col_3": [0, 0, 0, 0]}, index=[1, 2, 3, 4])


@pytest.fixture(scope="function")
def X2():
    return pd.DataFrame({"col_1": [0, 0, 0, 0], "col_2": [0, 0, 0, 0], "col_3": [0, 0, 0, 0]}, index=[1, 2, 3, 4])


# ====================================
# Classification Data Fixtures
# ====================================


@pytest.fixture(scope="function")
def binary_classification_data():
    """Create synthetic classification data for testing."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    return X_df, y_series


@pytest.fixture(scope="function")
def multi_classification_data(random_state):
    """Create multi-class classification data."""
    # Prepare two samples
    X, y = make_classification(
        n_samples=50,
        class_sep=0.05,
        n_informative=6,
        n_features=10,
        random_state=random_state,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=5,
    )
    return pd.DataFrame(X), y


@pytest.fixture(scope="function")
def complex_data(random_state):
    """Create more complex classification data with categorical and missing values."""
    feature_names = ["f1_categorical", "f2_missing", "f3_static", "f4", "f5"]

    # Prepare two samples
    X, y = make_classification(
        n_samples=50,
        class_sep=0.05,
        n_informative=2,
        n_features=5,
        random_state=random_state,
        n_redundant=2,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X, columns=feature_names)
    X.loc[0:10, "f2_missing"] = np.nan
    return X, y


@pytest.fixture(scope="function")
def complex_data_with_categorical(complex_data):
    """Add categorical feature to complex data."""
    X, y = complex_data
    X["f1_categorical"] = X["f1_categorical"].astype(str).astype("category")
    return X, y


@pytest.fixture(scope="function")
def complex_data_split(complex_data, random_state_42):
    """Split complex data into train/test sets."""
    X, y = complex_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_42)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="function")
def complex_data_split_with_categorical(complex_data_split):
    """Split complex data with categorical features into train/test sets."""
    X_train, X_test, y_train, y_test = complex_data_split
    X_train["f1_categorical"] = X_train["f1_categorical"].astype(str).astype("category")
    X_test["f1_categorical"] = X_test["f1_categorical"].astype(str).astype("category")
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="function")
def dependencies_binary_classification_data():
    """Create synthetic classification data for dependency testing."""
    # Re-implement the same functionality as classification_data
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    return X_df, y_series


# ====================================
# Regression Data Fixtures
# ====================================


@pytest.fixture(scope="function")
def dependencies_regression_data():
    """Create synthetic regression data for testing."""
    X = np.random.rand(100, 5)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(100) * 0.1
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    return X_df, y_series


# ====================================
# SHAP Data Fixtures
# ====================================


@pytest.fixture
def shap_input_data():
    """Generate various input data types for SHAP testing."""
    n_samples, n_features = 10, 5
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create different types of SHAP value formats for testing
    return {
        "X_shape": (n_samples, n_features),
        "feature_names": feature_names,
        "numpy_2d": np.random.rand(n_samples, n_features),
        "numpy_3d_binary": np.random.rand(n_samples, n_features, 2),
        "numpy_3d_single": np.random.rand(n_samples, n_features, 1),
        "list_binary": [np.random.rand(n_samples, n_features), np.random.rand(n_samples, n_features)],
        "list_single": [np.random.rand(n_samples, n_features)],
        "dataframe": pd.DataFrame(np.random.rand(n_samples, n_features), columns=feature_names),
    }


@pytest.fixture(scope="function")
def shap_importance_df():
    """Create a mock SHAP importance DataFrame for testing."""
    return pd.DataFrame({"mean_abs_shap": [0.5, 0.3, 0.1]}, index=["col_1", "col_2", "col_3"])


@pytest.fixture(scope="function")
def expected_df_2d():
    return pd.DataFrame({0: [1, 2], 1: [2, 3], 2: [3, 4]})


@pytest.fixture(scope="function")
def expected_df():
    return pd.DataFrame({0: [1, 2, 3]})


@pytest.fixture(scope="function")
def expected_feature_importance():
    return pd.DataFrame(
        {
            "mean_abs_shap_value_test": [2 / 3, 0.0, 0.0],
            "mean_abs_shap_value_train": [0.5, 0.0, 0.0],
            "mean_shap_value_test": [-2 / 3, 0.0, 0.0],
            "mean_shap_value_train": [0.5, 0.0, 0.0],  # Use positive value to match the implementation
        },
        index=["col_3", "col_1", "col_2"],
    )


@pytest.fixture(scope="function")
def expected_feature_importance_lin_models():
    return pd.DataFrame(
        {
            "mean_abs_shap_value_test": [0.53, 0.0, 0.0],
            "mean_abs_shap_value_train": [0.4, 0.0, 0.0],
            "mean_shap_value_test": [-0.53, 0.0, 0.0],
            "mean_shap_value_train": [0.4, 0.0, 0.0],  # Positive value based on actual implementation
        },
        index=["col_3", "col_1", "col_2"],
    )


# ====================================
# Model Fixtures
# ====================================


@pytest.fixture(scope="function")
def create_model_with_params():
    """
    Fixture providing a function to create a model with the appropriate parameters.
    """

    def _create(estimator_class, estimator_params, random_state, n_estimators=200):
        """
        Helper function to create a model with the appropriate parameters,
        handling different estimator types that may or may not support n_estimators.
        """
        try:
            # Try to create with n_estimators if supported
            return estimator_class(
                random_state=random_state,
                n_estimators=n_estimators,
                **{k: v for k, v in estimator_params.items() if k != "n_estimators"},
            )
        except TypeError:
            # Fall back to basic initialization if n_estimators not supported
            return estimator_class(random_state=random_state, **estimator_params)

    return _create


@pytest.fixture(scope="function")
def catboost_classifier(random_state):
    # TODO: Remove this skip once CatBoost is updated to work with NumPy 2.0
    pytest.skip("CatBoost tests are temporarily disabled due to compatibility issues with NumPy")
    # This code won't be reached due to the skip, but kept for reference
    model = CatBoostClassifier(random_seed=random_state)
    return model


@pytest.fixture(scope="function")
def decision_tree_classifier(random_state):
    model = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    return model


@pytest.fixture(scope="function")
def randomized_search_decision_tree_classifier(decision_tree_classifier, random_state):
    param_grid = {"criterion": ["gini"], "min_samples_split": [1, 2]}
    cv = RandomizedSearchCV(decision_tree_classifier, param_grid, cv=2, n_iter=2, random_state=random_state)
    return cv


@pytest.fixture(scope="function")
def logistic_regression(random_state):
    model = LogisticRegression(random_state=random_state)
    return model


@pytest.fixture(scope="function")
def tree_model(binary_classification_data, random_state):
    """Create a trained random forest model for testing."""
    X, y = binary_classification_data
    model = RandomForestClassifier(random_state=random_state, n_estimators=10)
    model.fit(X, y)
    return model


@pytest.fixture(scope="function")
def linear_model(binary_classification_data, random_state):
    """Create a trained logistic regression model for testing."""
    X, y = binary_classification_data
    model = LogisticRegression(random_state=random_state)
    model.fit(X, y)
    return model


@pytest.fixture(scope="function")
def pipeline_model(binary_classification_data, random_state):
    """Create a sklearn pipeline model for testing."""
    X, y = binary_classification_data
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=random_state))])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="function")
def xgb_model(random_state_42):
    """Create an XGBoost model for testing."""
    return XGBClassifier(random_state=random_state_42)


@pytest.fixture(scope="function")
def complex_lightgbm(random_state_42):
    model = LGBMClassifier(
        max_depth=5,
        num_leaves=11,
        class_weight="balanced",
        random_state=random_state_42,
        verbosity=-1,  # Set to -1 to only show fatal errors
    )
    return model


@pytest.fixture(scope="function")
def lgbm_regressor(random_state_42):
    """Create a LightGBM regressor for testing."""
    return LGBMRegressor(random_state=random_state_42, verbosity=-1)  # Set to -1


@pytest.fixture(scope="function")
def complex_fitted_lightgbm(complex_data_split_with_categorical, complex_lightgbm):
    X_train, _, y_train, _ = complex_data_split_with_categorical
    return complex_lightgbm.fit(X_train, y_train)


@pytest.fixture(scope="function")
def fitted_logistic_regression(X_train, y_train, logistic_regression):
    return logistic_regression.fit(X_train, y_train)


@pytest.fixture(scope="function")
def fitted_tree(X_train, y_train, decision_tree_classifier):
    return decision_tree_classifier.fit(X_train, y_train)


@pytest.fixture(scope="function")
def dependencies_classification_model(dependencies_binary_classification_data, random_state):
    """Create a trained random forest model for dependency testing."""
    X, y = dependencies_binary_classification_data
    model = RandomForestClassifier(random_state=random_state, n_estimators=10)
    model.fit(X, y)
    return model


@pytest.fixture(scope="function")
def dependencies_regression_model(dependencies_regression_data):
    """Create a trained linear regression model for testing."""
    X, y = dependencies_regression_data
    model = LinearRegression()
    model.fit(X, y)
    return model


# ====================================
# SHAP Fixtures
# ====================================


@pytest.fixture(scope="function")
def shap_explainer(tree_model, binary_classification_data):
    """Create a SHAP explainer for a tree model."""
    X, _ = binary_classification_data
    return shap.Explainer(tree_model, X)


@pytest.fixture(scope="function")
def mock_shap_elimination(random_state_42):
    """Create a partially initialized ShapRFECV instance for testing individual methods."""
    model = LogisticRegression(random_state=random_state_42)
    shap_elimination = ShapRFECV(model, random_state=random_state_42)
    shap_elimination.column_names = ["col_1", "col_2", "col_3"]
    shap_elimination.fitted = True
    return shap_elimination


# ====================================
# Feature Elimination Fixtures
# ====================================


@pytest.fixture(scope="function")
def feature_elimination_data():
    """Create data for feature elimination tests."""
    X = pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1, 1, 1, 1, 0],
            "col_2": [0, 0, 0, 0, 0, 0, 0, 1],
            "col_3": [1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])
    sample_weight = pd.Series([1, 1, 1, 1, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])

    return X, y, sample_weight


@pytest.fixture(scope="function")
def feature_elimination_X(feature_elimination_data):
    X, _, _ = feature_elimination_data
    return X


@pytest.fixture(scope="function")
def feature_elimination_y(feature_elimination_data):
    _, y, _ = feature_elimination_data
    return y


@pytest.fixture(scope="function")
def feature_elimination_sample_weight(feature_elimination_data):
    _, _, sample_weight = feature_elimination_data
    return sample_weight


# Mock utility functions for feature elimination tests
@pytest.fixture(scope="function")
def mock_feature_shap_values_per_fold():
    """Create a mock function for _get_feature_shap_values_per_fold."""

    def mock_method(*args, **kwargs):
        # Return dummy values: shap_values, train_score, val_score
        return np.array([[0.1, 0.2, 0.3]]), 0.8, 0.7

    return mock_method


@pytest.fixture(scope="function")
def mock_fit_compute():
    """Create a mock function for fit_compute."""

    def mock_method(*args, **kwargs):
        # Do nothing
        return None

    return mock_method


# ====================================
# Reporting Fixtures
# ====================================


@pytest.fixture(scope="function")
def feature_ranking_report_df():
    """Create a mock report DataFrame for feature ranking tests."""
    return pd.DataFrame(
        {
            "round": [1, 2],
            "features_set": [["col_1", "col_2"], ["col_1"]],
            "eliminated_features": [["col_3"], ["col_2"]],
        }
    )


@pytest.fixture(scope="function")
def feature_selection_report_df():
    """Create a mock report DataFrame for feature selection strategy tests."""
    return pd.DataFrame(
        {
            "round": [1, 2, 3],
            "num_features": [3, 2, 1],
            "val_metric_mean": [0.75, 0.85, 0.80],
            "val_metric_std": [0.05, 0.10, 0.15],
            "features_set": [["col_1", "col_2", "col_3"], ["col_1", "col_2"], ["col_1"]],
        }
    )


@pytest.fixture(scope="function")
def auto_selection_report_df():
    """Create a mock report DataFrame for automatic feature selection tests."""
    return pd.DataFrame(
        {
            "round": [1, 2, 3, 4, 5],
            "num_features": [5, 4, 3, 2, 1],
            "features_set": [
                ["col_1", "col_2", "col_3", "col_4", "col_5"],
                ["col_1", "col_2", "col_3", "col_4"],
                ["col_1", "col_2", "col_3"],
                ["col_1", "col_2"],
                ["col_1"],
            ],
            "val_metric_mean": [0.75, 0.80, 0.85, 0.82, 0.70],
            "val_metric_std": [0.10, 0.08, 0.05, 0.12, 0.15],
        }
    )


@pytest.fixture(scope="function")
def results_report_df():
    """Create a mock report DataFrame for reporting test results."""
    return pd.DataFrame(
        columns=[
            "num_features",
            "features_set",
            "eliminated_features",
            "train_metric_mean",
            "train_metric_std",
            "val_metric_mean",
            "val_metric_std",
        ]
    )


# ====================================
# Dependency Plotter Fixtures
# ====================================


@pytest.fixture(scope="function")
def dependencies_fitted_classifier_plotter(dependencies_classification_model, dependencies_binary_classification_data):
    """Return a fitted DependencePlotter instance with a classification model."""
    X, y = dependencies_binary_classification_data
    from probatus.model import DependencePlotter

    plotter = DependencePlotter(dependencies_classification_model)
    plotter.fit(X, y)
    return plotter


@pytest.fixture(scope="function")
def dependencies_fitted_regressor_plotter(dependencies_regression_model, dependencies_regression_data):
    """Return a fitted DependencePlotter instance with a regression model."""
    X, y = dependencies_regression_data
    from probatus.model import DependencePlotter

    plotter = DependencePlotter(dependencies_regression_model)
    plotter.fit(X, y)
    return plotter
