import pytest
import matplotlib
import matplotlib.pyplot as plt
import os
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from probatus.data_comparison.shap.importance import SHAPImportanceResemblance
from probatus.data_comparison.permutation.importance import PermutationImportanceResemblance
from probatus.model import ShapModelInterpreter, DependencePlotter
from probatus.selection import ShapRFECV
from probatus.selection._validation._parameters import _validate_model_compatibility_with_early_stopping_parameter

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")

# Define base plots directory path
BASE_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "multiclass_classification")

# Define estimators for parametrization
# TODO: Add CatBoostClassifier (when it supports NumPy 2.0)
ESTIMATORS = [
    pytest.param(
        LGBMClassifier,
        {"n_estimators": 100, "max_depth": 3, "verbose": -1},
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
        {"C": 1.0, "solver": "saga", "max_iter": 1000},
        {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2", "elasticnet"]},
        id="logistic_regression",
    ),
]


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_sample_similarity(
    multiclass_dataset,
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
    Test different estimators with SHAPImportanceResemblance for sample similarity analysis
    using multiclass data.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = multiclass_dataset
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Split data into two samples - class 0 vs class 1
    X1 = X[y == 0].reset_index(drop=True)
    X2 = X[y == 1].reset_index(drop=True)

    # Create model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)

    # Initialize resemblance model
    resemblance = SHAPImportanceResemblance(model=model, test_prc=0.3, n_jobs=1, verbose=1, random_state=random_state)

    # Fit and compute importance
    importance_df = resemblance.fit_compute(X1=X1, X2=X2, class_names=["Class 0", "Class 1"])

    # Verify results
    assert resemblance.class_names == ["Class 0", "Class 1"]
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
        assert check_plots_are_generated_correctly(
            plot_path
        ), "Sample similarity plot doesn't have enough colors - it may be empty or only showing axes."

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_permutation_importance_resemblance(
    multiclass_dataset,
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
    Test different estimators with PermutationImportanceResemblance for sample similarity analysis
    using multiclass data.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = multiclass_dataset
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Split data into two samples - class 0 vs class 2
    X1 = X[y == 0].reset_index(drop=True)
    X2 = X[y == 2].reset_index(drop=True)

    # Create model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)

    # Initialize resemblance model with PermutationImportanceResemblance
    resemblance = PermutationImportanceResemblance(
        model=model, iterations=20, test_prc=0.3, n_jobs=1, verbose=1, random_state=random_state
    )

    # Fit and compute importance
    importance_df = resemblance.fit_compute(X1=X1, X2=X2, class_names=["Class 0", "Class 2"])

    # Verify results
    assert resemblance.class_names == ["Class 0", "Class 2"]
    assert importance_df.shape[0] == X.shape[1]
    # The score might be exactly 0.5 if the samples are not distinguishable
    assert resemblance.train_score >= 0.5
    assert resemblance.test_score >= 0.5

    # Test plotting and save the plot
    fig = resemblance.plot(show=False)
    assert fig is not None

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_permutation_importance.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(
            plot_path
        ), "Permutation importance plot doesn't have enough colors - it may be empty or only showing axes."

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_model_interpret(
    multiclass_dataset,
    split_dataset,
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
    Test different estimators with ShapModelInterpreter for model interpretation
    using multiclass data.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(multiclass_dataset)
    class_names = [f"Class {i}" for i in range(3)]  # Iris has 3 classes
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create and fit model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)
    model.fit(X_train, y_train)

    # Initialize model interpreter
    shap_interpret = ShapModelInterpreter(model, verbose=1, scoring="roc_auc_ovo", random_state=random_state)

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
    fig = shap_interpret.plot("summary", target_set="test", show=False)
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_model_interpret_importance.png")
        # Use fig.savefig since we're working with a Figure object
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(
            plot_path
        ), "Model interpret importance plot doesn't have enough colors - it may be empty or only showing axes."

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_shap_dependence(
    multiclass_dataset,
    split_dataset,
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
    Test different estimators with DependencePlotter for SHAP dependence plots
    using multiclass data.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X_train, X_test, y_train, y_test = split_dataset(multiclass_dataset)
    class_names = [f"Class {i}" for i in range(3)]  # Iris has 3 classes
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create and fit model with the specified estimator class and parameters
    model = estimator_class(random_state=random_state, **estimator_params)
    model.fit(X_train, y_train)

    # Initialize DependencePlotter
    dependence_plotter = DependencePlotter(model, verbose=1, random_state=random_state)

    # Fit the plotter directly
    dependence_plotter.fit(X_test, y_test, class_names=class_names, approximate=False, check_additivity=False)

    # Test dependence plots for single numeric feature (using Iris feature names)
    numeric_feature = "sepal length (cm)"
    fig_result = dependence_plotter.plot(feature=numeric_feature, show=False)
    assert fig_result is not None
    assert isinstance(fig_result, plt.Figure)

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_dependence_numeric.png")
        fig_result.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(
            plot_path
        ), "SHAP dependence numeric plot doesn't have enough colors - it may be empty or only showing axes."

    plt.close("all")

    # Test with multiple numeric features (using Iris feature names)
    numeric_features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]

    for i, feature in enumerate(numeric_features):
        fig = dependence_plotter.plot(feature=feature, show=False)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Save the plot if save_plots is True
        if save_plots:
            plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_dependence_multiple_{i}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(
                plot_path
            ), f"SHAP dependence multiple plot {i} doesn't have enough colors - it may be empty or only showing axes."

    # Test with custom quantile range and alpha parameters
    fig = dependence_plotter.plot(feature="sepal length (cm)", min_q=0.1, max_q=0.9, alpha=0.7, show=False)
    assert fig is not None
    assert isinstance(fig, plt.Figure)

    # Save the plot if save_plots is True
    if save_plots:
        plot_path = os.path.join(plots_dir, f"{estimator_name}_shap_dependence_custom_params.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Verify the plot has diverse colors
        assert check_plots_are_generated_correctly(
            plot_path
        ), "SHAP dependence with custom parameters plot doesn't have enough colors - it may be empty or only showing axes."

    # Close all plots to free memory
    plt.close("all")


@pytest.mark.parametrize("estimator_class, estimator_params, param_grid", ESTIMATORS)
def test_feature_elimination_randomized_search_early_stopping(
    multiclass_dataset,
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
    Test different estimators with RandomizedSearchCV and early stopping for feature elimination,
    using the multiclass_dataset fixture.
    """
    # Create plot directories if save_plots is True
    setup_plot_dirs(save_plots, BASE_PLOTS_DIR, ESTIMATORS)

    X, y = multiclass_dataset
    plots_dir = get_plots_dir(BASE_PLOTS_DIR, estimator_class, ESTIMATORS)
    estimator_name = next(param.id for param in ESTIMATORS if param.values[0] == estimator_class)

    # Create base model with the specified estimator class
    model = create_model_with_params(estimator_class, estimator_params, random_state)

    # Create RandomizedSearchCV with the appropriate param_grid
    search = RandomizedSearchCV(model, param_grid, cv=2, n_iter=2, random_state=random_state)
    is_compatible = _validate_model_compatibility_with_early_stopping_parameter(search)

    # Set the appropriate eval_metric based on the estimator type
    if estimator_class == LGBMClassifier:
        eval_metric_value = "multi_logloss"
    elif estimator_class == XGBClassifier:
        eval_metric_value = "mlogloss"
    else:
        eval_metric_value = "multi_logloss"  # Default for other compatible estimators

    # For non-compatible estimators, expect ValueError during initialization
    if not is_compatible:
        with pytest.raises(TypeError, match="Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping"):
            ShapRFECV(
                search,
                step=1,
                cv=3,
                scoring="roc_auc_ovo",
                early_stopping_rounds=5,
                eval_metric=eval_metric_value,  # This value doesn't matter for non-compatible estimators
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
            scoring="roc_auc_ovo",
            early_stopping_rounds=5,
            eval_metric=eval_metric_value,
            n_jobs=1,
            verbose=1,
            random_state=random_state,
        )

        # Fit and compute feature importance
        report = shap_elimination.fit_compute(X, y)

        # Verify results
        assert report.shape[0] == X.shape[1]
        assert len(shap_elimination.get_optimal_feature_selection(1)) == 1

        # Test plotting and save the plot
        fig = shap_elimination.plot(show=False)
        assert fig is not None

        # Save the plot if save_plots is True
        if save_plots:
            plot_path = os.path.join(plots_dir, f"{estimator_name}_feature_elimination_randomized_search.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Verify the plot has diverse colors
            assert check_plots_are_generated_correctly(
                plot_path
            ), "Feature elimination with randomized search plot doesn't have enough colors - it may be empty or only showing axes."

        # Close all plots to free memory
        plt.close("all")
