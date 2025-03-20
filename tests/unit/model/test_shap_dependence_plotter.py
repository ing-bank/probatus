import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from probatus.model.shap_dependence_plotter import DependencePlotter
from probatus.core import NotFittedError


def test_dependence_plotter_fit(dependencies_classification_model, dependencies_binary_classification_data):
    """Test fitting the plotter with a classification model."""
    X, y = dependencies_binary_classification_data
    plotter = DependencePlotter(dependencies_classification_model)

    # Test basic fit
    fitted_plotter = plotter.fit(X, y)
    assert fitted_plotter is plotter  # Returns self
    assert plotter.fitted is True
    assert hasattr(plotter, "shap_values")
    # Shap values can be a DataFrame or a numpy array
    assert isinstance(plotter.shap_values, (np.ndarray, pd.DataFrame))
    assert plotter.shap_values.shape == X.shape

    # Test fit with precalculated SHAP values
    precalc_shap = pd.DataFrame(np.random.rand(*X.shape), columns=X.columns)
    plotter = DependencePlotter(dependencies_classification_model)
    plotter.fit(X, y, precalc_shap=precalc_shap)
    if isinstance(plotter.shap_values, np.ndarray):
        assert np.array_equal(plotter.shap_values, precalc_shap.values)
    else:
        assert plotter.shap_values.equals(precalc_shap)


def test_class_names_integration(dependencies_classification_model, dependencies_binary_classification_data):
    """Test the integration of class_names handling in fit and plotting methods."""
    X, y = dependencies_binary_classification_data

    # Create test data with multi-class scenario (3 classes)
    multi_y = pd.Series(np.random.choice([0, 1, 2], size=len(y)))

    # Test with None (default labels)
    plotter = DependencePlotter(dependencies_classification_model)
    plotter.fit(X, multi_y)
    assert plotter.class_names == ["label_0", "label_1", "label_2"]

    # Test with list of class names
    plotter = DependencePlotter(dependencies_classification_model)
    plotter.fit(X, multi_y, class_names=["Ape", "Lion", "Bear"])
    assert plotter.class_names == ["Ape", "Lion", "Bear"]

    # Test with dictionary of class names
    plotter = DependencePlotter(dependencies_classification_model)
    plotter.fit(X, multi_y, class_names={1: "Ape", 2: "Lion", 0: "Bear"})
    assert plotter.class_names == ["Bear", "Ape", "Lion"]

    # Test with string keys in dictionary
    plotter = DependencePlotter(dependencies_classification_model)
    plotter.fit(X, multi_y, class_names={"0": "Ape", "2": "Bear", "1": "Lion"})
    assert plotter.class_names == ["Ape", "Lion", "Bear"]


def test_dependence_plotter_plot(dependencies_fitted_classifier_plotter, dependencies_binary_classification_data):
    """Test plot method creates a figure with expected components."""
    X, _ = dependencies_binary_classification_data

    # Test plot with feature name string
    fig = dependencies_fitted_classifier_plotter.plot(feature="feature_0", show=False)
    assert isinstance(fig, plt.Figure)
    # There are 3 axes in the classification case (main plot, histogram, and target rate)
    assert len(fig.axes) == 3

    # Test plot with feature index
    fig = dependencies_fitted_classifier_plotter.plot(feature=1, show=False)
    assert isinstance(fig, plt.Figure)

    # Test plot with custom figsize and bins
    fig = dependencies_fitted_classifier_plotter.plot(feature="feature_0", figsize=(10, 8), bins=5, show=False)
    assert fig.get_size_inches()[0] == 10
    assert fig.get_size_inches()[1] == 8

    # Test plot with min_q, max_q, and alpha
    fig = dependencies_fitted_classifier_plotter.plot(feature="feature_0", min_q=0.1, max_q=0.9, alpha=0.5, show=False)
    assert dependencies_fitted_classifier_plotter.min_q == 0.1
    assert dependencies_fitted_classifier_plotter.max_q == 0.9
    assert dependencies_fitted_classifier_plotter.alpha == 0.5

    # Test plot with invalid feature name
    with pytest.raises(ValueError, match="not recognized"):
        dependencies_fitted_classifier_plotter.plot(feature="non_existent_feature")

    # Test plot with invalid feature index
    with pytest.raises(ValueError, match="out of range"):
        dependencies_fitted_classifier_plotter.plot(feature=100)

    # Test plot with invalid min_q, max_q
    with pytest.raises(ValueError, match="min_q must be smaller than max_q"):
        dependencies_fitted_classifier_plotter.plot(feature="feature_0", min_q=0.9, max_q=0.1)

    # Test plot with invalid alpha
    with pytest.raises(ValueError, match="alpha must be a float value between 0 and 1"):
        dependencies_fitted_classifier_plotter.plot(feature="feature_0", alpha=2.0)


def test_dependence_plotter_target_rate_plot(dependencies_fitted_classifier_plotter):
    """Test _target_rate_plot internal method."""
    bin_edges, fig, target_ratio = dependencies_fitted_classifier_plotter._target_rate_plot(feature="feature_0", bins=5)
    assert isinstance(bin_edges, (list, np.ndarray))
    assert isinstance(fig, plt.Figure)
    assert isinstance(target_ratio, pd.Series)
    assert len(bin_edges) == 6  # 5 bins = 6 edges

    # Test with list of bin edges
    custom_bins = [-np.inf, -1, 0, 1, np.inf]
    bin_edges, fig, target_ratio = dependencies_fitted_classifier_plotter._target_rate_plot(
        feature="feature_0", bins=custom_bins
    )
    assert len(bin_edges) == len(custom_bins)

    # Test with invalid bin type
    with pytest.raises(TypeError, match="bins must be an integer or a list of floats"):
        dependencies_fitted_classifier_plotter._target_rate_plot(feature="feature_0", bins=3.5)


def test_dependence_plotter_get_X_y_shap_with_q_cut(
    dependencies_fitted_classifier_plotter, dependencies_binary_classification_data
):
    """Test _get_X_y_shap_with_q_cut internal method."""
    X, _ = dependencies_binary_classification_data
    feature = "feature_0"

    # Test with default quantile range (0, 1)
    x, y, shap_val = dependencies_fitted_classifier_plotter._get_X_y_shap_with_q_cut(feature=feature)
    assert isinstance(x, pd.Series)
    assert isinstance(y, pd.Series)
    assert isinstance(shap_val, (pd.Series, np.ndarray))
    assert len(x) == len(X)  # All data points included

    # Test with custom quantile range
    dependencies_fitted_classifier_plotter.min_q = 0.25
    dependencies_fitted_classifier_plotter.max_q = 0.75
    x, y, shap_val = dependencies_fitted_classifier_plotter._get_X_y_shap_with_q_cut(feature=feature)
    assert len(x) < len(X)  # Some data points excluded

    # Test with invalid feature
    with pytest.raises(ValueError, match="not found in data"):
        dependencies_fitted_classifier_plotter._get_X_y_shap_with_q_cut(feature="invalid_feature")

    # Test with not fitted plotter
    plotter = DependencePlotter(dependencies_fitted_classifier_plotter.model)
    with pytest.raises(NotFittedError, match="not fitted yet"):
        plotter._get_X_y_shap_with_q_cut(feature=feature)
