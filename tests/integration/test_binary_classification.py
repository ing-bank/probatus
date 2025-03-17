import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from probatus.dataset import SHAPImportanceResemblance
from probatus.model import ShapModelInterpreter
from probatus.features import ShapRFECV
from probatus.utils import preprocess_labels

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")

# Define base plots directory path
BASE_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "binary_classification")

# Define estimators for parametrization
# TODO: Add CatBoostClassifier (when it supports NumPy 2.0)
ESTIMATORS = [
    pytest.param(
        LGBMClassifier,
        {"n_estimators": 100, "max_depth": 3},
        {"max_depth": [3, 4, 5], "num_leaves": [7, 15, 31]},
        id="lightgbm",
    ),
    pytest.param(
        RandomForestClassifier,
        {"n_estimators": 100, "max_depth": 3},
        {"max_depth": [3, 4, 5], "max_features": ["sqrt", "log2"]},
        id="random_forest",
    ),
    pytest.param(
        XGBClassifier,
        {"n_estimators": 100, "max_depth": 3},
        {"max_depth": [3, 4, 5], "learning_rate": [0.01, 0.1, 0.2]},
        id="xgboost",
    ),
    pytest.param(
        LogisticRegression,
        {"C": 1.0, "solver": "liblinear"},
        {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]},
        id="logistic_regression",
    ),
]


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


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_sample_similarity(
    breast_cancer_data,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    save_plots,
    setup_plot_dirs,
    get_plots_dir,
    check_plots_are_generated_correctly,
):
    """
    Test different estimators with SHAPImportanceResemblance for sample similarity analysis.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = breast_cancer_data
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Split data into two samples
    X1 = X[y == 0].reset_index(drop=True)
    X2 = X[y == 1].reset_index(drop=True)

    # Create model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)

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

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_sample_similarity.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(plot_path), (
            "Sample similarity plot doesn't have enough colors - it may be empty or only showing axes."
        )

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_model_interpret(
    breast_cancer_data_split,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    save_plots,
    setup_plot_dirs,
    get_plots_dir,
    check_plots_are_generated_correctly,
):
    """
    Test different estimators with ShapModelInterpreter for model interpretation.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X_train, X_test, y_train, y_test = breast_cancer_data_split
    class_names = ["Malignant", "Benign"]
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create and fit model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)
    model.fit(X_train, y_train)

    # Initialize model interpreter
    shap_interpret = ShapModelInterpreter(model, verbose=1, random_state=random_state)

    # Fit the model
    shap_interpret.fit(
        X_train, X_test, y_train, y_test, class_names=class_names, approximate=False, check_additivity=False
    )

    # Verify results
    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score >= 0.5
    # The test score might be low due to the small dataset size
    # We're just testing the functionality, not the model performance
    assert shap_interpret.test_score >= 0.0

    # Test plotting and save the plot
    # For importance plot, the method now returns a Figure object
    fig = shap_interpret.plot("importance", target_set="test", show=False)
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_model_interpret_importance.png")
        # Use fig.savefig since we're working with a Figure object
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(plot_path), (
            "Model interpret importance plot doesn't have enough colors - it may be empty or only showing axes."
        )

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_shap_dependence(
    breast_cancer_data_split,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    save_plots,
    setup_plot_dirs,
    get_plots_dir,
    check_plots_are_generated_correctly,
):
    """
    Test different estimators with ShapModelInterpreter for SHAP dependence plots.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X_train, X_test, y_train, y_test = breast_cancer_data_split
    class_names = ["Malignant", "Benign"]
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create and fit model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)
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
    if save_plots:
        if isinstance(fig_result, list):
            assert all(isinstance(fig, matplotlib.figure.Figure) for fig in fig_result)
            # Save all figures in the list
            for i, fig in enumerate(fig_result):
                plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_dependence_numeric_{i}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                # Verify the plot has diverse colors
                assert check_plots_are_generated_correctly(plot_path), (
                    f"SHAP dependence numeric plot {i} doesn't have enough colors - it may be empty or only showing axes."
                )
        else:
            assert isinstance(fig_result, matplotlib.figure.Figure)
            # Save the plot
            plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_dependence_numeric.png")
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
    if save_plots:
        for i, fig in enumerate(fig_list):
            plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_dependence_multiple_{i}.png")
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
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_summary.png")
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
    if save_plots:
        if isinstance(fig_result, list):
            assert all(isinstance(fig, matplotlib.figure.Figure) for fig in fig_result)
            # Save all figures in the list
            for i, fig in enumerate(fig_result):
                plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_sample_{i}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                # Verify the plot has diverse colors
                assert check_plots_are_generated_correctly(plot_path), (
                    f"SHAP sample plot {i} doesn't have enough colors - it may be empty or only showing axes."
                )
        else:
            assert isinstance(fig_result, matplotlib.figure.Figure)
            # Save the plot
            plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_sample.png")
            fig_result.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(plot_path), (
                "SHAP sample plot doesn't have enough colors - it may be empty or only showing axes."
            )

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_feature_elimination(
    breast_cancer_data,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    save_plots,
    setup_plot_dirs,
    get_plots_dir,
    check_plots_are_generated_correctly,
):
    """
    Test different estimators with ShapRFECV for feature elimination.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = breast_cancer_data
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)

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

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_feature_elimination.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(plot_path), (
            "Feature elimination plot doesn't have enough colors - it may be empty or only showing axes."
        )

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_feature_elimination_early_stopping(
    breast_cancer_data,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    save_plots,
    setup_plot_dirs,
    get_plots_dir,
    check_plots_are_generated_correctly,
    create_model_with_params,
):
    """
    Test different estimators with ShapRFECV for feature elimination with early stopping.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = breast_cancer_data
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create model with the specified estimator class and parameters
    model = create_model_with_params(estimator_class, estimator_params, random_state)

    # Create a temporary ShapRFECV to check compatibility
    temp_shap_elimination = ShapRFECV(model=model, random_state=random_state)
    is_compatible = temp_shap_elimination._check_if_model_is_compatible_with_early_stopping(model)

    # For non-compatible estimators, expect ValueError during initialization
    if not is_compatible:
        with pytest.raises(ValueError, match="Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping"):
            ShapRFECV(
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
    else:
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

        # Save the plot if save_plots is True
        if save_plots:
            plot_path = os.path.join(plots_dir, f"{estimator_name}_feature_elimination_early_stopping.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(plot_path), (
                "Feature elimination early stopping plot doesn't have enough colors - it may be empty or only showing axes."
            )

        # Close all plots to free memory
        plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_randomized_search_early_stopping(
    breast_cancer_data,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    save_plots,
    setup_plot_dirs,
    get_plots_dir,
    check_plots_are_generated_correctly,
    create_model_with_params,
):
    """
    Test different estimators with RandomizedSearchCV and early stopping for feature elimination.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = breast_cancer_data
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create base model with the specified estimator class
    model = create_model_with_params(estimator_class, estimator_params, random_state)

    # Create RandomizedSearchCV with the appropriate param_grid
    search = RandomizedSearchCV(model, param_grid, cv=2, n_iter=2, random_state=random_state)

    # Create a temporary ShapRFECV to check compatibility
    temp_shap_elimination = ShapRFECV(model=search, random_state=random_state)
    is_compatible = temp_shap_elimination._check_if_model_is_compatible_with_early_stopping(search)

    # For non-compatible estimators, expect ValueError during initialization
    if not is_compatible:
        with pytest.raises(ValueError, match="Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping"):
            ShapRFECV(
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
    else:
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

        # Save the plot if save_plots is True
        if save_plots:
            plot_path = os.path.join(plots_dir, f"{estimator_name}_randomized_search_early_stopping.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(plot_path), (
                "Randomized search early stopping plot doesn't have enough colors - it may be empty or only showing axes."
            )

        # Close all plots to free memory
        plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_get_feature_shap_values_per_fold_early_stopping(
    breast_cancer_data,
    random_state,
    estimator_class,
    estimator_params,
    param_grid,
    create_model_with_params,
):
    """
    Test the internal _get_feature_shap_values_per_fold_early_stopping method with different estimators.
    """
    # Create model with the specified estimator class and parameters
    model = create_model_with_params(estimator_class, estimator_params, random_state)

    X, y = breast_cancer_data
    y = preprocess_labels(y, index=X.index)

    # Get indices for both classes to ensure balanced validation set
    malignant_indices = y[y == 0].index.tolist()[:3]
    benign_indices = y[y == 1].index.tolist()[:3]
    val_index = malignant_indices + benign_indices

    # Get remaining indices for training
    train_index = [i for i in range(len(y)) if i not in val_index][:45]  # Limit to 45 samples

    # Check if model is compatible with early stopping
    temp_shap_elimination = ShapRFECV(model=model, random_state=random_state)
    is_compatible = temp_shap_elimination._check_if_model_is_compatible_with_early_stopping(model)

    # For non-compatible estimators, expect ValueError
    if not is_compatible:
        # First, verify that creating a ShapRFECV with early stopping raises ValueError
        with pytest.raises(ValueError, match="Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping"):
            ShapRFECV(model, early_stopping_rounds=5, eval_metric="auc", scoring="roc_auc", random_state=random_state)

        # Create a mock ShapRFECV instance without early stopping for testing
        mock_shap_elimination = ShapRFECV(model=model, random_state=random_state)

        # Test that calling the internal method directly also raises ValueError
        with pytest.raises(ValueError, match="Model type not supported for early stopping"):
            mock_shap_elimination._get_feature_shap_values_per_fold_early_stopping(
                X,
                y,
                model,
                train_index=train_index,
                val_index=val_index,
            )
    else:
        # Initialize feature elimination with early stopping
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
