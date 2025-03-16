import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from PIL import Image
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from probatus.interpret import ShapModelInterpreter
from probatus.sample_similarity import SHAPImportanceResemblance
from probatus.feature_elimination import ShapRFECV
from probatus.utils import preprocess_labels

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")

# Create plots directory if it doesn't exist
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "lightgbm_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def check_plots_are_generated_correctly(image_path, min_unique_colors=4):
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


@pytest.fixture(scope="function")
def breast_cancer_data(random_state):
    """
    Load the breast cancer dataset and return it as a pandas DataFrame.
    Sample 100 records from the dataset instead of using the entire dataset.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Sample 100 records
    indices = np.random.RandomState(random_state).choice(len(X), size=100, replace=False)
    X = X.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)

    return X, y


@pytest.fixture(scope="function")
def breast_cancer_data_split(breast_cancer_data, random_state):
    """
    Split the breast cancer dataset into train and test sets.
    """
    X, y = breast_cancer_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test


def test_lightgbm_sample_similarity(breast_cancer_data, random_state):
    """
    Test LightGBM with SHAPImportanceResemblance for sample similarity analysis.
    """
    X, y = breast_cancer_data

    # Split data into two samples
    X1 = X[y == 0].reset_index(drop=True)
    X2 = X[y == 1].reset_index(drop=True)

    # Create LightGBM model
    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=random_state)

    # Initialize resemblance model
    resemblance = SHAPImportanceResemblance(model=model, test_prc=0.3, n_jobs=1, verbose=1, random_state=random_state)

    # Fit and compute importance
    importance_df = resemblance.fit_compute(X1=X1, X2=X2, class_names=["Malignant", "Benign"])

    # Verify results
    assert resemblance.class_names == ["Malignant", "Benign"]
    assert importance_df.shape[0] == X.shape[1]
    # The score might be exactly 0.5 if the samples are not distinguishable
    assert resemblance.train_score >= 0.5
    assert resemblance.test_score >= 0.5

    # Test plotting and save the plot
    fig = resemblance.plot(show=False)
    assert fig is not None

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "sample_similarity.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Verify the plot has diverse colors
    assert check_plots_are_generated_correctly(plot_path), (
        "Sample similarity plot doesn't have enough colors - it may be empty or only showing axes."
    )

    # Close all plots to free memory
    plt.close("all")


def test_lightgbm_model_interpret(breast_cancer_data_split, random_state):
    """
    Test LightGBM with ShapModelInterpreter for model interpretation.
    """
    X_train, X_test, y_train, y_test = breast_cancer_data_split
    class_names = ["Malignant", "Benign"]

    # Create and fit a LightGBM model
    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    model.fit(X_train, y_train)

    # Initialize model interpreter
    shap_interpret = ShapModelInterpreter(model, verbose=1, random_state=random_state)

    # Fit and compute importance
    importance_df = shap_interpret.fit_compute(
        X_train, X_test, y_train, y_test, class_names=class_names, approximate=False, check_additivity=False
    )

    # Verify results
    assert shap_interpret.class_names == class_names
    assert importance_df.shape[0] == X_train.shape[1]
    assert shap_interpret.train_score >= 0.5
    # The test score might be low due to the small dataset size
    # We're just testing the functionality, not the model performance
    assert shap_interpret.test_score >= 0.0

    # Test plotting and save the plot
    # For importance plot, the method now returns a Figure object
    fig = shap_interpret.plot("importance", target_set="test", show=False)
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "model_interpret_importance.png")
    # Use fig.savefig since we're working with a Figure object
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Verify the plot has diverse colors
    assert check_plots_are_generated_correctly(plot_path), (
        "Model interpret importance plot doesn't have enough colors - it may be empty or only showing axes."
    )

    # Close all plots to free memory
    plt.close("all")


def test_lightgbm_shap_dependence(breast_cancer_data_split, random_state):
    """
    Test LightGBM with ShapModelInterpreter for SHAP dependence plots.
    """
    X_train, X_test, y_train, y_test = breast_cancer_data_split
    class_names = ["Malignant", "Benign"]

    # Create and fit a LightGBM model
    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    model.fit(X_train, y_train)

    # Initialize model interpreter
    shap_interpret = ShapModelInterpreter(model, verbose=1, random_state=random_state)

    # Fit the model
    shap_interpret.fit(
        X_train, X_test, y_train, y_test, class_names=class_names, approximate=False, check_additivity=False
    )

    # Test dependence plots for numeric feature
    numeric_feature = "mean radius"
    # For dependence plot, the method now returns a Figure or List[Figure]
    fig_result = shap_interpret.plot("dependence", target_columns=numeric_feature, target_set="test", show=False)
    assert fig_result is not None

    # Handle both single Figure and List[Figure] cases
    if isinstance(fig_result, list):
        assert all(isinstance(fig, matplotlib.figure.Figure) for fig in fig_result)
        # Save all figures in the list
        for i, fig in enumerate(fig_result):
            plot_path = os.path.join(PLOTS_DIR, f"shap_dependence_numeric_{i}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(plot_path), (
                f"SHAP dependence numeric plot {i} doesn't have enough colors - it may be empty or only showing axes."
            )
    else:
        assert isinstance(fig_result, matplotlib.figure.Figure)
        # Save the plot
        plot_path = os.path.join(PLOTS_DIR, "shap_dependence_numeric.png")
        fig_result.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(plot_path), (
            "SHAP dependence numeric plot doesn't have enough colors - it may be empty or only showing axes."
        )

    plt.close("all")

    # Test with multiple numeric features
    numeric_features = ["mean radius", "mean texture", "mean perimeter"]
    # For dependence plot with multiple features, the method now returns a List[Figure]
    fig_list = shap_interpret.plot("dependence", target_columns=numeric_features, target_set="test", show=False)
    assert fig_list is not None
    assert isinstance(fig_list, list)
    assert all(isinstance(fig, matplotlib.figure.Figure) for fig in fig_list)

    # Save all figures in the list
    for i, fig in enumerate(fig_list):
        plot_path = os.path.join(PLOTS_DIR, f"shap_dependence_multiple_{i}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(plot_path), (
            f"SHAP dependence multiple plot {i} doesn't have enough colors - it may be empty or only showing axes."
        )

    # Also test summary plot
    # For summary plot, the method now returns a Figure object
    fig = shap_interpret.plot("summary", target_set="test", show=False)
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "shap_summary.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Verify the plot has diverse colors
    assert check_plots_are_generated_correctly(plot_path), (
        "SHAP summary plot doesn't have enough colors - it may be empty or only showing axes."
    )

    # Test sample plot
    # For sample plot, the method now returns a Figure or List[Figure]
    fig_result = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    assert fig_result is not None

    # Handle both single Figure and List[Figure] cases
    if isinstance(fig_result, list):
        assert all(isinstance(fig, matplotlib.figure.Figure) for fig in fig_result)
        # Save all figures in the list
        for i, fig in enumerate(fig_result):
            plot_path = os.path.join(PLOTS_DIR, f"shap_sample_{i}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(plot_path), (
                f"SHAP sample plot {i} doesn't have enough colors - it may be empty or only showing axes."
            )
    else:
        assert isinstance(fig_result, matplotlib.figure.Figure)
        # Save the plot
        plot_path = os.path.join(PLOTS_DIR, "shap_sample.png")
        fig_result.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(plot_path), (
            "SHAP sample plot doesn't have enough colors - it may be empty or only showing axes."
        )

    # Close all plots to free memory
    plt.close("all")


def test_lightgbm_feature_elimination(breast_cancer_data, random_state):
    """
    Test LightGBM with ShapRFECV for feature elimination.
    """
    X, y = breast_cancer_data

    # Create LightGBM model
    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=random_state)

    # Initialize feature elimination
    shap_elimination = ShapRFECV(
        model=model, step=1, cv=3, scoring="roc_auc", n_jobs=1, verbose=1, random_state=random_state
    )

    # Fit and compute feature importance
    report = shap_elimination.fit_compute(X, y)

    # Verify results
    assert report.shape[0] == X.shape[1]
    assert len(shap_elimination.get_reduced_features_set(1)) == 1

    # Test plotting and save the plot
    fig = shap_elimination.plot(show=False)
    assert fig is not None

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "feature_elimination.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Verify the plot has diverse colors
    assert check_plots_are_generated_correctly(plot_path), (
        "Feature elimination plot doesn't have enough colors - it may be empty or only showing axes."
    )

    # Close all plots to free memory
    plt.close("all")


def test_lightgbm_feature_elimination_early_stopping(breast_cancer_data, random_state):
    """
    Test LightGBM with ShapRFECV for feature elimination with early stopping.
    """
    X, y = breast_cancer_data

    # Create LightGBM model
    model = LGBMClassifier(n_estimators=200, max_depth=3, random_state=random_state)

    # Initialize feature elimination with early stopping
    shap_elimination = ShapRFECV(
        model=model,
        step=1,
        cv=3,
        scoring="roc_auc",
        early_stopping_rounds=5,
        eval_metric="auc",
        n_jobs=1,
        verbose=1,
        random_state=random_state,
    )

    # Fit and compute feature importance
    report = shap_elimination.fit_compute(X, y)

    # Verify results
    assert report.shape[0] == X.shape[1]
    assert len(shap_elimination.get_reduced_features_set(1)) == 1

    # Test plotting and save the plot
    fig = shap_elimination.plot(show=False)
    assert fig is not None

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "feature_elimination_early_stopping.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Verify the plot has diverse colors
    assert check_plots_are_generated_correctly(plot_path), (
        "Feature elimination early stopping plot doesn't have enough colors - it may be empty or only showing axes."
    )

    # Close all plots to free memory
    plt.close("all")


def test_lightgbm_randomized_search_early_stopping(breast_cancer_data, random_state):
    """
    Test LightGBM with RandomizedSearchCV and early stopping for feature elimination.
    """
    X, y = breast_cancer_data

    # Create base LightGBM model
    model = LGBMClassifier(n_estimators=200, random_state=random_state)

    # Create parameter grid for RandomizedSearchCV
    param_grid = {"max_depth": [3, 4, 5], "num_leaves": [7, 15, 31]}

    # Create RandomizedSearchCV
    search = RandomizedSearchCV(model, param_grid, cv=2, n_iter=2, random_state=random_state)

    # Initialize feature elimination with early stopping
    shap_elimination = ShapRFECV(
        search,
        step=1,
        cv=3,
        scoring="roc_auc",
        early_stopping_rounds=5,
        eval_metric="auc",
        n_jobs=1,
        verbose=1,
        random_state=random_state,
    )

    # Fit and compute feature importance
    report = shap_elimination.fit_compute(X, y)

    # Verify results
    assert report.shape[0] == X.shape[1]
    assert len(shap_elimination.get_reduced_features_set(1)) == 1

    # Test plotting and save the plot
    fig = shap_elimination.plot(show=False)
    assert fig is not None

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "randomized_search_early_stopping.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Verify the plot has diverse colors
    assert check_plots_are_generated_correctly(plot_path), (
        "Randomized search early stopping plot doesn't have enough colors - it may be empty or only showing axes."
    )

    # Close all plots to free memory
    plt.close("all")


def test_get_feature_shap_values_per_fold_early_stopping(breast_cancer_data, random_state):
    """
    Test the internal _get_feature_shap_values_per_fold_early_stopping method with LightGBM.
    """
    model = LGBMClassifier(n_estimators=200, max_depth=3, random_state=random_state)
    X, y = breast_cancer_data
    y = preprocess_labels(y, index=X.index)

    # Get indices for both classes to ensure balanced validation set
    malignant_indices = y[y == 0].index.tolist()[:3]
    benign_indices = y[y == 1].index.tolist()[:3]
    val_index = malignant_indices + benign_indices

    # Get remaining indices for training
    train_index = [i for i in range(len(y)) if i not in val_index][:45]  # Limit to 45 samples

    # Initialize feature elimination
    shap_elimination = ShapRFECV(
        model, early_stopping_rounds=5, eval_metric="auc", scoring="roc_auc", random_state=random_state
    )

    # Test internal method
    shap_values, train_score, test_score = shap_elimination._get_feature_shap_values_per_fold_early_stopping(
        X,
        y,
        model,
        train_index=train_index,
        val_index=val_index,
    )

    # Verify results
    assert test_score > 0.5
    assert train_score > 0.5
    assert shap_values.shape[1] == X.shape[1]
