import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import os
import tempfile
from matplotlib.figure import Figure

from probatus.interpret import ShapModelInterpreter


@pytest.fixture(scope="function")
def expected_feature_importance():
    return pd.DataFrame(
        {
            "mean_abs_shap_value_test": [0.5, 0.0, 0.0],
            "mean_abs_shap_value_train": [0.5, 0.0, 0.0],
            "mean_shap_value_test": [-0.5, 0.0, 0.0],
            "mean_shap_value_train": [-0.5, 0.0, 0.0],
        },
        index=["col_3", "col_1", "col_2"],
    )


@pytest.fixture(scope="function")
def expected_feature_importance_lin_models():
    return pd.DataFrame(
        {
            "mean_abs_shap_value_test": [0.4, 0.0, 0.0],
            "mean_abs_shap_value_train": [0.4, 0.0, 0.0],
            "mean_shap_value_test": [-0.4, 0.0, 0.0],
            "mean_shap_value_train": [-0.4, 0.0, 0.0],
        },
        index=["col_3", "col_1", "col_2"],
    )


def test_shap_interpret(fitted_tree, X_train, y_train, X_test, y_test, expected_feature_importance, random_state):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)
    assert (np.mean(np.abs(shap_interpret.shap_values_test), axis=0) == [0, 0, 0.5]).all()
    assert (np.mean(np.abs(shap_interpret.shap_values_train), axis=0) == [0, 0, 0.5]).all()

    importance_df, train_auc, test_auc = shap_interpret.compute(return_scores=True)

    pd.testing.assert_frame_equal(expected_feature_importance, importance_df)
    assert train_auc == 1
    assert test_auc == pytest.approx(0.833, 0.01)

    # Check if plots work for such dataset
    fig1 = shap_interpret.plot("importance", target_set="test", show=False)
    fig2 = shap_interpret.plot("summary", target_set="test", show=False)
    fig3 = shap_interpret.plot("dependence", target_columns="col_3", target_set="test", show=False)
    fig4 = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    fig5 = shap_interpret.plot("importance", target_set="train", show=False)
    fig6 = shap_interpret.plot("summary", target_set="train", show=False)
    fig7 = shap_interpret.plot("dependence", target_columns="col_3", target_set="train", show=False)
    fig8 = shap_interpret.plot("sample", samples_index=X_train.index.tolist()[0:2], target_set="train", show=False)

    # Verify return types
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
    assert isinstance(fig3, list) and len(fig3) == 2 and all(isinstance(fig, Figure) for fig in fig3)
    assert isinstance(fig4, list) and len(fig4) == 2 and all(isinstance(fig, Figure) for fig in fig4)
    assert isinstance(fig5, Figure)
    assert isinstance(fig6, Figure)
    assert isinstance(fig7, list) and len(fig7) == 2 and all(isinstance(fig, Figure) for fig in fig7)
    assert isinstance(fig8, list) and len(fig8) == 2 and all(isinstance(fig, Figure) for fig in fig8)

    # Close all plots to free memory
    plt.close("all")


def test_shap_interpret_lin_models(
    fitted_logistic_regression, X_train, y_train, X_test, y_test, expected_feature_importance_lin_models, random_state
):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_logistic_regression, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_test), axis=0), 2) == [0, 0, 0.4]).all()
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_train), axis=0), 2) == [0, 0, 0.4]).all()

    importance_df, train_auc, test_auc = shap_interpret.compute(return_scores=True)
    importance_df = importance_df.round(2)

    pd.testing.assert_frame_equal(expected_feature_importance_lin_models, importance_df)
    assert train_auc == 1
    assert test_auc == pytest.approx(0.833, 0.01)

    # Check if plots work for such dataset
    fig1 = shap_interpret.plot("importance", target_set="test", show=False)
    fig2 = shap_interpret.plot("summary", target_set="test", show=False)
    fig3 = shap_interpret.plot("dependence", target_columns="col_3", target_set="test", show=False)
    fig4 = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    fig5 = shap_interpret.plot("importance", target_set="train", show=False)
    fig6 = shap_interpret.plot("summary", target_set="train", show=False)
    fig7 = shap_interpret.plot("dependence", target_columns="col_3", target_set="train", show=False)
    fig8 = shap_interpret.plot("sample", samples_index=X_train.index.tolist()[0:2], target_set="train", show=False)

    # Verify return types
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
    assert isinstance(fig3, list) and len(fig3) == 2 and all(isinstance(fig, Figure) for fig in fig3)
    assert isinstance(fig4, list) and len(fig4) == 2 and all(isinstance(fig, Figure) for fig in fig4)
    assert isinstance(fig5, Figure)
    assert isinstance(fig6, Figure)
    assert isinstance(fig7, list) and len(fig7) == 2 and all(isinstance(fig, Figure) for fig in fig7)
    assert isinstance(fig8, list) and len(fig8) == 2 and all(isinstance(fig, Figure) for fig in fig8)

    # Close all plots to free memory
    plt.close("all")


def test_shap_interpret_fit_compute_lin_models(
    fitted_logistic_regression, X_train, y_train, X_test, y_test, expected_feature_importance_lin_models, random_state
):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_logistic_regression, random_state=random_state)
    importance_df = shap_interpret.fit_compute(X_train, X_test, y_train, y_test, class_names=class_names)
    importance_df = importance_df.round(2)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1

    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_test), axis=0), 2) == [0, 0, 0.4]).all()
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_train), axis=0), 2) == [0, 0, 0.4]).all()

    pd.testing.assert_frame_equal(expected_feature_importance_lin_models, importance_df)


def test_shap_interpret_fit_compute(
    fitted_tree, X_train, y_train, X_test, y_test, expected_feature_importance, random_state
):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    importance_df = shap_interpret.fit_compute(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.mean(np.abs(shap_interpret.shap_values_test), axis=0) == [0, 0, 0.5]).all()
    assert (np.mean(np.abs(shap_interpret.shap_values_train), axis=0) == [0, 0, 0.5]).all()

    pd.testing.assert_frame_equal(expected_feature_importance, importance_df)


def test_shap_interpret_complex_data(complex_data_split_with_categorical, complex_fitted_lightgbm, random_state):
    class_names = ["neg", "pos"]
    X_train, X_test, y_train, y_test = complex_data_split_with_categorical

    shap_interpret = ShapModelInterpreter(complex_fitted_lightgbm, verbose=1, random_state=random_state)
    importance_df = shap_interpret.fit_compute(
        X_train, X_test, y_train, y_test, class_names=class_names, approximate=False, check_additivity=False
    )

    assert shap_interpret.class_names == class_names
    assert importance_df.shape[0] == X_train.shape[1]

    # Check if plots work for such dataset
    fig1 = shap_interpret.plot("importance", target_set="test", show=False)
    fig2 = shap_interpret.plot("summary", target_set="test", show=False)
    fig3 = shap_interpret.plot("dependence", target_columns="f2_missing", target_set="test", show=False)
    fig4 = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    fig5 = shap_interpret.plot("importance", target_set="train", show=False)
    fig6 = shap_interpret.plot("summary", target_set="train", show=False)
    fig7 = shap_interpret.plot("dependence", target_columns="f2_missing", target_set="train", show=False)
    fig8 = shap_interpret.plot("sample", samples_index=X_train.index.tolist()[0:2], target_set="train", show=False)

    # Verify return types
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
    assert isinstance(fig3, list) and len(fig3) == 2 and all(isinstance(fig, Figure) for fig in fig3)
    assert isinstance(fig4, list) and len(fig4) == 2 and all(isinstance(fig, Figure) for fig in fig4)
    assert isinstance(fig5, Figure)
    assert isinstance(fig6, Figure)
    assert isinstance(fig7, list) and len(fig7) == 2 and all(isinstance(fig, Figure) for fig in fig7)
    assert isinstance(fig8, list) and len(fig8) == 2 and all(isinstance(fig, Figure) for fig in fig8)

    # Close all plots to free memory
    plt.close("all")


def test_shap_interpret_waterfall_plot(fitted_tree, X_train, y_train, X_test, y_test, random_state):
    """
    Test the waterfall plot functionality in ShapModelInterpreter.plot() method.
    """
    class_names = ["neg", "pos"]

    # Initialize and fit the ShapModelInterpreter
    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    # Test waterfall plot with a single test sample
    single_test_sample = X_test.index.tolist()[0]
    fig1 = shap_interpret.plot("sample", samples_index=single_test_sample, target_set="test", show=False)

    # Test waterfall plot with multiple test samples
    multiple_test_samples = X_test.index.tolist()[0:2]
    fig2 = shap_interpret.plot("sample", samples_index=multiple_test_samples, target_set="test", show=False)

    # Test waterfall plot with a single train sample
    single_train_sample = X_train.index.tolist()[0]
    fig3 = shap_interpret.plot("sample", samples_index=single_train_sample, target_set="train", show=False)

    # Test waterfall plot with multiple train samples
    multiple_train_samples = X_train.index.tolist()[0:2]
    fig4 = shap_interpret.plot("sample", samples_index=multiple_train_samples, target_set="train", show=False)

    # Verify the return types of the plots
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, list) and len(fig2) == 2 and all(isinstance(fig, Figure) for fig in fig2)
    assert isinstance(fig3, Figure)
    assert isinstance(fig4, list) and len(fig4) == 2 and all(isinstance(fig, Figure) for fig in fig4)

    # Close all plots to free memory
    plt.close("all")


def test_shap_interpret_multiclass(random_state):
    """
    Test that ShapModelInterpreter correctly handles multiclass models and generates appropriate plots.
    """
    # Create a multiclass dataset
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Create a directory to save the plots
    plot_dir = os.path.join(tempfile.gettempdir(), "probatus_multiclass_plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to: {plot_dir}")

    # Create a synthetic multiclass dataset with 3 classes
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=random_state
    )

    # Convert to pandas DataFrame/Series
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    # Create and train a multiclass model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_state)
    model.fit(X_train, y_train)

    # Define class names
    class_names = ["Class_0", "Class_1", "Class_2"]

    # Initialize and fit the ShapModelInterpreter with multiclass ROC AUC scoring
    shap_interpret = ShapModelInterpreter(model, scoring="roc_auc_ovr", random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    # Verify that class names are set correctly
    assert shap_interpret.class_names == class_names

    # Verify that SHAP values have been calculated correctly
    # For multiclass, SHAP values should have the right dimensions
    assert shap_interpret.shap_values_train is not None
    assert shap_interpret.shap_values_test is not None

    # Compute feature importance
    importance_df = shap_interpret.compute()

    # Verify importance DataFrame has the expected structure
    assert isinstance(importance_df, pd.DataFrame)
    assert importance_df.shape[0] == 4  # Should have one row per feature
    assert all(
        col in importance_df.columns
        for col in [
            "mean_abs_shap_value_test",
            "mean_abs_shap_value_train",
            "mean_shap_value_test",
            "mean_shap_value_train",
        ]
    )

    # Verify that all features are present in the importance DataFrame
    assert set(importance_df.index) == set(feature_names)

    # Test all plot types to ensure they work with multiclass models
    # Importance plot
    fig1 = shap_interpret.plot("importance", target_set="test", show=True)
    assert isinstance(fig1, Figure)

    # Save the figure
    fig1.savefig(os.path.join(plot_dir, "multiclass_importance_plot.png"), dpi=300, bbox_inches="tight")

    # Verify importance plot content
    # Get the main axes from the figure
    ax1 = fig1.axes[0]
    # Check that the plot title contains the expected text
    assert "SHAP Feature Importance" in ax1.get_title()
    # Check that the plot contains the expected number of bars (one per feature)
    assert len(ax1.patches) == len(feature_names)
    # Check that the x-axis label is appropriate
    assert "mean(|SHAP value|)" in ax1.get_xlabel()

    # Summary plot
    fig2 = shap_interpret.plot("summary", target_set="test", show=True)
    assert isinstance(fig2, Figure)

    # Save the figure
    fig2.savefig(os.path.join(plot_dir, "multiclass_summary_plot.png"), dpi=300, bbox_inches="tight")

    # Verify summary plot content
    # Get the main axes from the figure
    ax2 = fig2.axes[0]
    # Check that the plot title contains the expected text
    assert "SHAP Summary" in ax2.get_title()
    # Check that the x-axis label is appropriate
    assert "SHAP value" in ax2.get_xlabel()
    # Check that the y-axis contains all features
    y_tick_labels = [label.get_text() for label in ax2.get_yticklabels()]
    assert len(y_tick_labels) == len(feature_names)

    # Dependence plot for a specific feature
    fig_result3 = shap_interpret.plot("dependence", target_columns="feature_1", target_set="test", show=True)
    assert isinstance(fig_result3, list)
    assert len(fig_result3) == 2  # Should return 2 plots for dependence
    assert all(isinstance(fig, Figure) for fig in fig_result3)

    # Save the dependence plots
    for i, fig in enumerate(fig_result3):
        fig.savefig(os.path.join(plot_dir, f"multiclass_dependence_plot_{i}.png"), dpi=300, bbox_inches="tight")

    # Check that the plot titles contain the expected feature name
    # The title might be in a different axes or might not be set as expected
    # Let's check if any of the axes has the expected feature name
    feature_name_found = False
    for fig in fig_result3:
        for ax in fig.axes:
            if ax.get_title() and "feature_1" in ax.get_title():
                feature_name_found = True
                break
        if feature_name_found:
            break
    assert feature_name_found, "Could not find feature_1 in any axes title"

    # Check that the y-axis label mentions SHAP values
    shap_label_found = False
    for fig in fig_result3:
        for ax in fig.axes:
            if ax.get_ylabel() and ("SHAP value" in ax.get_ylabel() or "Shap value" in ax.get_ylabel()):
                shap_label_found = True
                break
        if shap_label_found:
            break
    assert shap_label_found, "Could not find SHAP value in any axes y-label"

    # Sample plot for a specific sample
    sample_index = X_test.index[0]
    fig4 = shap_interpret.plot("sample", samples_index=sample_index, target_set="test", show=True)
    assert isinstance(fig4, Figure)

    # Save the figure
    fig4.savefig(os.path.join(plot_dir, "multiclass_sample_plot.png"), dpi=300, bbox_inches="tight")

    # Check that the plot title contains the expected text
    # The title might be in a different axes or might not be set as expected
    # Let's check if any of the axes has the expected title
    title_found = False
    index_found = False
    for ax in fig4.axes:
        if ax.get_title() and "SHAP" in ax.get_title() and "sample" in ax.get_title().lower():
            title_found = True
        if ax.get_title() and str(sample_index) in ax.get_title():
            index_found = True
        if title_found and index_found:
            break
    assert title_found, "Could not find SHAP sample explanation title in any axes"
    assert index_found, f"Could not find sample index {sample_index} in any axes title"

    # Multiple sample plots
    sample_indices = X_test.index[:2]
    fig_result5 = shap_interpret.plot("sample", samples_index=sample_indices, target_set="test", show=True)
    assert isinstance(fig_result5, list)
    assert len(fig_result5) == 2  # Should return one plot per sample
    assert all(isinstance(fig, Figure) for fig in fig_result5)

    # Save the multiple sample plots
    for i, fig in enumerate(fig_result5):
        fig.savefig(os.path.join(plot_dir, f"multiclass_multiple_sample_plot_{i}.png"), dpi=300, bbox_inches="tight")

    # Verify multiple sample plots content
    for i, fig in enumerate(fig_result5):
        title_found = False
        index_found = False
        for ax in fig.axes:
            if ax.get_title() and "SHAP" in ax.get_title() and "sample" in ax.get_title().lower():
                title_found = True
            if ax.get_title() and str(sample_indices[i]) in ax.get_title():
                index_found = True
            if title_found and index_found:
                break
        assert title_found, "Could not find SHAP sample explanation title in any axes"
        assert index_found, f"Could not find sample index {sample_indices[i]} in any axes title"

    # Test with different plot parameters
    # Test importance plot with max_display parameter
    fig6 = shap_interpret.plot("importance", target_set="test", show=True, max_display=2)
    assert isinstance(fig6, Figure)

    # Save the figure
    fig6.savefig(os.path.join(plot_dir, "multiclass_importance_plot_max_display.png"), dpi=300, bbox_inches="tight")

    # Should only display 2 features
    ax6 = fig6.axes[0]
    assert len(ax6.patches) == 2

    # Test summary plot with different plot_type
    # The plot_type is already determined by the first parameter ("summary"),
    # so we can't pass it again as a keyword argument
    # Instead, let's test with a different parameter like max_display
    fig7 = shap_interpret.plot("summary", target_set="test", show=True, max_display=2)
    assert isinstance(fig7, Figure)

    # Save the figure
    fig7.savefig(os.path.join(plot_dir, "multiclass_summary_plot_max_display.png"), dpi=300, bbox_inches="tight")

    # Test dependence plot with different interaction_index
    # The DependencePlotter.plot method doesn't accept interaction_index parameter
    # Let's use parameters that are actually supported by the method
    fig_result8 = shap_interpret.plot(
        "dependence", target_columns="feature_1", target_set="test", show=True, bins=5, alpha=0.7
    )
    assert isinstance(fig_result8, list)
    assert len(fig_result8) == 2
    assert all(isinstance(fig, Figure) for fig in fig_result8)

    # Save the custom dependence plots
    for i, fig in enumerate(fig_result8):
        fig.savefig(os.path.join(plot_dir, f"multiclass_dependence_plot_custom_{i}.png"), dpi=300, bbox_inches="tight")

    # Close all plots to free memory
    plt.close("all")
