import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Global constant seed for reproducibility
SEED = 42
DATASET_SIZE = 100

# ====================================
# Random State Fixtures
# ====================================


@pytest.fixture(scope="session")
def random_state():
    """Return the global seed for reproducibility."""
    return SEED


# ====================================
# Dataset Fixtures
# ====================================


@pytest.fixture(scope="session")
def binary_classification_dataset():
    """
    Load the breast cancer dataset for binary classification testing.

    This is a real dataset limited to 100 samples and 30 features.
    The task is to classify tumors as malignant or benign.

    Returns:
        tuple: (X, y) where X is a pandas DataFrame and y is a pandas Series
    """
    data = load_breast_cancer(as_frame=True)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Sample 100 records
    indices = np.random.RandomState(SEED).choice(len(X), size=DATASET_SIZE, replace=False)
    X = X.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)

    return X, y


@pytest.fixture(scope="session")
def multiclass_dataset():
    """
    Load the iris dataset for multiclass classification testing.

    This is a real dataset limited to 100 samples and 4 features.
    The task is to classify iris plants into 3 different species.

    Returns:
        tuple: (X, y) where X is a pandas DataFrame and y is a pandas Series
    """
    data = load_iris(as_frame=True)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Sample 100 records
    indices = np.random.RandomState(SEED).choice(len(X), size=DATASET_SIZE, replace=False)
    X = X.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)

    return X, y


@pytest.fixture(scope="session")
def regression_dataset():
    """
    Load the diabetes dataset for regression testing.

    This is a real dataset limited to 100 samples and 10 features.
    The task is to predict a quantitative measure of disease progression.

    Returns:
        tuple: (X, y) where X is a pandas DataFrame and y is a pandas Series
    """
    data = load_diabetes(as_frame=True)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Sample 100 records
    indices = np.random.RandomState(SEED).choice(len(X), size=DATASET_SIZE, replace=False)
    X = X.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)

    return X, y


@pytest.fixture
def split_dataset():
    """
    Fixture to split any dataset into train and test sets.

    Usage:
        def test_something(split_dataset, binary_classification_dataset):
            X_train, X_test, y_train, y_test = split_dataset(binary_classification_dataset)
            # or with custom test_size
            X_train, X_test, y_train, y_test = split_dataset(binary_classification_dataset, test_size=0.3)

    Args:
        dataset_fixture: One of binary_classification_dataset, multi-class_dataset or regression_dataset
        test_size: Proportion of the dataset to include in the test split (default: 0.2)

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where all are pandas objects
    """

    def _split_dataset(dataset, test_size=0.2):
        X, y = dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
        return X_train, X_test, y_train, y_test

    return _split_dataset


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
# Binary Classification Fixtures
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
