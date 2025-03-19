import numpy as np
import pandas as pd
import pytest
import shap
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

from probatus.utils.shap import (
    _validate_shap_inputs,
    _create_shap_explainer,
    _compute_shap_values,
    _format_shap_values,
    calculate_shap_explanation,
    _shap_values_to_df,
    calculate_shap_importance,
    shap_explanation_to_shap_df,
)


@pytest.mark.parametrize(
    "model_fixture, expected_valid, expected_error_contains",
    [
        ("tree_model", True, None),
        ("linear_model", True, None),
        ("pipeline_model", False, "Pipeline"),
    ],
)
def test_validate_shap_inputs(
    request, binary_classification_data, model_fixture, expected_valid, expected_error_contains
):
    """Test _validate_shap_inputs with various model types."""
    model = request.getfixturevalue(model_fixture)
    X, _ = binary_classification_data

    # Test validation
    is_valid, error_message = _validate_shap_inputs(model, X, verbose=0)
    assert is_valid is expected_valid

    if expected_error_contains:
        assert expected_error_contains in error_message


def test_validate_shap_inputs_non_dataframe(tree_model, binary_classification_data):
    """Test _validate_shap_inputs with non-DataFrame input."""
    X, _ = binary_classification_data
    X_array = X.values

    with pytest.warns(UserWarning, match="not a pandas DataFrame"):
        _validate_shap_inputs(tree_model, X_array, verbose=1)


@pytest.mark.parametrize(
    "model_fixture, input_type",
    [
        ("tree_model", "dataframe"),
        ("linear_model", "dataframe"),
        ("tree_model", "array_with_categorical"),
        ("tree_model", "small_dataset"),
    ],
)
def test_create_shap_explainer(request, binary_classification_data, model_fixture, input_type):
    """Test _create_shap_explainer with different models and data types."""
    model = request.getfixturevalue(model_fixture)
    X, _ = binary_classification_data

    # Handle different input types
    if input_type == "dataframe":
        input_data = X
    elif input_type == "array_with_categorical":
        input_data = pd.DataFrame(
            {
                "num1": np.random.rand(50),
                "num2": np.random.rand(50),
                "cat1": pd.Series(np.random.choice(["A", "B", "C"], 50)).astype("category"),
            }
        )
    elif input_type == "small_dataset":
        input_data = pd.DataFrame(np.random.rand(10, 3), columns=["f1", "f2", "f3"])

    # Test explainer creation
    explainer = _create_shap_explainer(model, input_data, random_state=42)
    assert isinstance(explainer, shap.Explainer)


@pytest.mark.parametrize(
    "approximate, check_additivity, dataset_type",
    [
        # (True, True, "binary"),
        (False, True, "multiclass"),
        # (True, False, "regression"),
    ],
)
def test_compute_shap_values(request, approximate, check_additivity, dataset_type):
    """
    Test computing SHAP values with different dataset types, checking output shapes.

    Uses multi_classification_data as base dataset, modified appropriately for each type:
    - binary: filtered to keep only 2 classes
    - multiclass: used as is (multiple classes)
    - regression: same features but with continuous target
    """
    # Get the multi-class data as base
    X, y = request.getfixturevalue("multi_classification_data")

    # Prepare data and model based on dataset type
    if dataset_type == "binary":
        # Keep only the first two classes to create binary data
        binary_indices = np.where(y < 2)[0]
        X_binary = X.iloc[binary_indices].reset_index(drop=True)
        y_binary = y[binary_indices]
        y_binary = (y_binary == 1).astype(int)  # Convert to 0/1 target

        # Train binary classifier
        model = RandomForestClassifier(random_state=42, n_estimators=5)
        model.fit(X_binary, y_binary)

        # Set test data and expected dimensions
        test_X = X_binary

    elif dataset_type == "multiclass":
        # Use multiclass data as is
        # Train multiclass classifier
        model = RandomForestClassifier(random_state=42, n_estimators=5)
        model.fit(X, y)

        # Set test data and expected dimensions
        test_X = X

    elif dataset_type == "regression":
        # Create continuous target for regression
        # (Use a simple formula based on features to create synthetic regression target)
        from sklearn.ensemble import RandomForestRegressor

        continuous_y = X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.normal(0, 0.5, len(X))

        # Train regression model
        model = RandomForestRegressor(random_state=42, n_estimators=5)
        model.fit(X, continuous_y)

        # Set test data and expected dimensions
        test_X = X

    # Create a real explainer for this test (more realistic test)
    explainer = _create_shap_explainer(model, test_X, random_state=42)

    # Call function being tested
    shap_explanation = _compute_shap_values(
        explainer=explainer, X=test_X, approximate=approximate, check_additivity=check_additivity
    )

    # Verify output shape matches expectations
    assert shap_explanation.values.shape[0] == test_X.shape[0]
    assert shap_explanation.values.shape[1] == test_X.shape[1]
    if dataset_type == "multiclass":
        assert shap_explanation.values.shape[2] == len(np.unique(y))


@pytest.mark.parametrize(
    "multiclass_aggregation, feature",
    [
        (None, None),  # Default behavior
        ("max_abs", None),  # Max absolute aggregation
        ("variance", None),  # Variance aggregation
        ("mean_abs", None),  # Mean absolute aggregation
        (None, 0),  # Specific class by index
    ],
)
def test_format_shap_values_multiclass(multiclass_aggregation, feature):
    """Test _format_shap_values with multiclass data and different aggregation methods."""
    # Create a mock Explanation object with multiclass data
    mock_explanation = MagicMock()
    n_samples, n_features, n_classes = 10, 5, 3
    mock_explanation.values = np.random.rand(n_samples, n_features, n_classes)
    mock_explanation.output_names = [0, 1, 2]  # Class names

    # Format the SHAP values
    formatted = _format_shap_values(
        shap_explanation=mock_explanation, feature=feature, multiclass_aggregation=multiclass_aggregation
    )

    # Check that the output has the correct shape
    assert isinstance(formatted, np.ndarray)
    assert formatted.shape[0] == n_samples
    assert formatted.shape[1] == n_features

    # Check specific aggregation logic
    if multiclass_aggregation == "max_abs":
        # Check that values match max absolute across classes
        expected = np.max(np.abs(mock_explanation.values), axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif multiclass_aggregation == "variance":
        # Check that values match variance across classes
        expected = np.var(mock_explanation.values, axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif multiclass_aggregation == "mean_abs":
        # Check that values match mean absolute across classes
        expected = np.mean(np.abs(mock_explanation.values), axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif feature is not None:
        # Check that values match the specified class
        class_idx = mock_explanation.output_names.index(feature)
        expected = mock_explanation.values[:, :, class_idx]
        np.testing.assert_array_equal(formatted, expected)
    else:
        # Default behavior should select first class
        expected = mock_explanation.values[:, :, 0]
        np.testing.assert_array_equal(formatted, expected)


def test_format_shap_values_invalid_feature():
    """Test _format_shap_values with an invalid feature/class name."""
    # Create a mock Explanation object
    mock_explanation = MagicMock()
    mock_explanation.values = np.random.rand(10, 5, 3)
    mock_explanation.output_names = [0, 1, 2]  # Class names

    # Try to format with an invalid feature name
    with pytest.raises(ValueError, match="Feature 3 not found"):
        _format_shap_values(mock_explanation, feature=3)


@pytest.mark.parametrize(
    "model_fixture, sample_size, check_additivity, return_explainer",
    [
        ("tree_model", 100, False, False),
        ("linear_model", 50, True, False),
        ("tree_model", 20, False, True),
    ],
)
def test_calculate_shap_explanation(
    request, binary_classification_data, model_fixture, sample_size, check_additivity, return_explainer
):
    """Test calculate_shap_explanation with different arguments."""
    model = request.getfixturevalue(model_fixture)
    X, _ = binary_classification_data

    # Calculate SHAP explanation
    result = calculate_shap_explanation(
        model=model,
        X=X,
        return_explainer=return_explainer,
        sample_size=sample_size,
        check_additivity=check_additivity,
        random_state=42,
    )

    if return_explainer:
        shap_explanation, explainer = result
        assert isinstance(explainer, shap.Explainer)
    else:
        shap_explanation = result

    assert isinstance(shap_explanation, shap.Explanation)


def test_shap_calc_with_pipeline(pipeline_model, binary_classification_data):
    """Test that using a Pipeline raises an error."""
    X, _ = binary_classification_data

    with pytest.raises(TypeError, match="Pipeline"):
        calculate_shap_explanation(pipeline_model, X)


@pytest.mark.parametrize(
    "multiclass_aggregation",
    [
        None,  # Default behavior
        "max_abs",  # Max absolute aggregation
        "variance",  # Variance aggregation
        "mean_abs",  # Mean absolute aggregation
    ],
)
def test_shap_explanation_to_shap_df(tree_model, binary_classification_data, multiclass_aggregation):
    """Test shap_explanation_to_shap_df function."""
    X, _ = binary_classification_data

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=tree_model, X=X, return_explainer=False, random_state=42)

    # Convert to DataFrame with different aggregation methods
    shap_df = shap_explanation_to_shap_df(
        shap_explanation=shap_explanation, model=tree_model, X=X, multiclass_aggregation=multiclass_aggregation
    )

    # Check output
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)
    assert list(shap_df.index) == list(X.index)


@pytest.mark.parametrize(
    "input_type, has_precalc, feature, multiclass_aggregation",
    [
        ("dataframe", True, None, None),
        ("dataframe", False, None, "max_abs"),
        ("numpy", True, None, "variance"),
        ("numpy", False, None, "mean_abs"),
        ("empty", False, None, None),
    ],
)
def test_shap_values_to_df(
    tree_model, binary_classification_data, shap_input_data, input_type, has_precalc, feature, multiclass_aggregation
):
    """Test _shap_values_to_df with different input types and options."""
    X, _ = binary_classification_data

    if input_type == "dataframe":
        input_data = X
        shape = X.shape
    elif input_type == "numpy":
        input_data = X.values
        shape = X.shape
    elif input_type == "empty":
        input_data = pd.DataFrame()

    if input_type == "empty":
        with pytest.raises(ValueError, match="cannot be empty"):
            _shap_values_to_df(tree_model, input_data)
    else:
        if has_precalc:
            precalc_shap = np.random.rand(*shape)
            shap_df = _shap_values_to_df(tree_model, input_data, precalc_shap)
        else:
            # Test with the additional parameters
            shap_df = _shap_values_to_df(
                model=tree_model, X=input_data, feature=feature, multiclass_aggregation=multiclass_aggregation
            )

        assert isinstance(shap_df, pd.DataFrame)
        assert shap_df.shape == shape

        if input_type == "numpy":
            # For numpy arrays, column names should be integers starting from 0
            assert all(isinstance(col, int) for col in shap_df.columns)
        else:
            # For dataframes, column names should match original columns
            assert list(shap_df.columns) == list(X.columns)


@pytest.mark.parametrize(
    "input_type, columns_arg, suffix, variance_penalty",
    [
        ("numpy", "feature_names", None, None),
        ("dataframe", None, None, None),
        ("numpy", "custom", None, None),
        ("dataframe", None, "_test", 0.5),
        ("numpy", "feature_names", "_model1", 0.8),  # Test with both suffix and penalty
        ("dataframe", None, "_comparison", -0.1),  # Test with negative penalty (should be ignored)
    ],
)
def test_calculate_shap_importance(shap_input_data, input_type, columns_arg, suffix, variance_penalty):
    """Test calculate_shap_importance with different input formats and options."""
    X_shape = shap_input_data["X_shape"]
    feature_names = shap_input_data["feature_names"]

    if input_type == "numpy":
        shap_values = shap_input_data["numpy_2d"]
    else:
        shap_values = shap_input_data["dataframe"]

    # Determine columns to use
    if columns_arg == "feature_names":
        columns = feature_names
    elif columns_arg == "custom":
        columns = [f"custom_{i}" for i in range(X_shape[1])]
    elif columns_arg is None and input_type == "dataframe":
        columns = shap_values.columns
    else:
        columns = None

    # Determine if suffix should be used
    kwargs = {}
    if suffix:
        kwargs["output_columns_suffix"] = suffix
    if variance_penalty is not None:
        kwargs["shap_variance_penalty_factor"] = variance_penalty

    # Calculate importance
    importance = calculate_shap_importance(shap_values, columns=columns, **kwargs)

    # Check results
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == X_shape[1]

    # Check column names based on whether suffix is provided
    base_column = "mean_abs_shap_value"
    expected_column = f"{base_column}{suffix or ''}"
    assert expected_column in importance.columns

    base_column = "mean_shap_value"
    expected_column = f"{base_column}{suffix or ''}"
    assert expected_column in importance.columns

    # Check specific options
    if columns_arg == "custom":
        assert set(importance.index) == {f"custom_{i}" for i in range(X_shape[1])}

    if variance_penalty is not None and variance_penalty > 0:
        penalized_column = f"penalized_mean_abs_shap_value{suffix or ''}"
        abs_shap_column = f"mean_abs_shap_value{suffix or ''}"
        assert penalized_column in importance.columns
        assert (importance[penalized_column] <= importance[abs_shap_column]).all()
    elif variance_penalty is not None and variance_penalty <= 0:
        # Negative or zero penalty should not result in penalized column
        penalized_column = f"penalized_mean_abs_shap_value{suffix or ''}"
        assert penalized_column not in importance.columns


def test_calculate_shap_importance_dimension_mismatch(shap_input_data):
    """Test that a dimension mismatch raises an appropriate error."""
    shap_values = shap_input_data["numpy_2d"]
    wrong_columns = [f"feature_{i}" for i in range(shap_values.shape[1] - 1)]  # One fewer column

    with pytest.raises(ValueError, match="Dimension mismatch"):
        calculate_shap_importance(shap_values, columns=wrong_columns)


@pytest.mark.parametrize(
    "shap_values_type",
    ["numpy_2d"],
)
def test_calculate_shap_importance_regular(shap_input_data, shap_values_type):
    """Test calculate_shap_importance with regular 2D SHAP values."""
    shap_values = shap_input_data[shap_values_type]
    feature_names = shap_input_data["feature_names"]

    # Calculate importance
    importance = calculate_shap_importance(shap_values, columns=feature_names)

    # Check results
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == shap_values.shape[1]  # Should have one row per feature
    assert "mean_abs_shap_value" in importance.columns
    assert "mean_shap_value" in importance.columns

    # Check that all feature names are in the index
    for feature in feature_names:
        assert feature in importance.index


def test_calculate_shap_importance_multiclass():
    """Test calculate_shap_importance with multiclass SHAP values."""
    # Create sample data with proper dimensions for multiclass
    n_samples, n_features, n_classes = 10, 5, 3
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create a properly shaped 3D array (n_samples, n_features, n_classes)
    # For multiclass, the format should be (n_classes, n_samples, n_features)
    # but our function expects (n_samples, n_features, n_classes)
    shap_values = np.random.rand(n_samples, n_features, n_classes)

    # The function calculate_shap_importance should handle this by taking the sum across classes
    with (
        patch("probatus.utils.shap.np.ndim") as mock_ndim,
        patch("probatus.utils.shap.np.sum") as mock_sum,
        patch("probatus.utils.shap.np.mean") as mock_mean,
        patch("probatus.utils.shap.np.std") as mock_std,
    ):
        # Set up mocks to return expected values
        mock_ndim.return_value = 3  # Multiclass

        # Mock the sum operation for multiclass. Should handle summing across the class dimension
        # and return data with shape (n_samples, n_features)
        sum_abs_shap = np.random.rand(n_samples, n_features)
        sum_shap = np.random.rand(n_samples, n_features)
        mock_sum.side_effect = [sum_abs_shap, sum_shap]

        # Mock means and std. Should be 1D arrays with n_features elements
        shap_abs_mean = np.random.rand(n_features)
        shap_mean = np.random.rand(n_features)
        std_vals = np.random.rand(n_features)
        mock_mean.side_effect = [shap_abs_mean, shap_mean]
        mock_std.return_value = std_vals

        # Calculate importance
        importance = calculate_shap_importance(shap_values, columns=feature_names)

        # Check results
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == n_features  # Should have one row per feature
        assert "mean_abs_shap_value" in importance.columns
        assert "mean_shap_value" in importance.columns

        # Check that all feature names are in the index
        for i, feature in enumerate(feature_names):
            assert feature in importance.index


@pytest.mark.parametrize("check_additivity", [True, False])
def test_shap_additivity_checking(tree_model, binary_classification_data, check_additivity):
    """Test that the check_additivity parameter is properly passed to the explainer."""
    X, _ = binary_classification_data

    # Use mock to check if the additivity check is performed
    with patch("probatus.utils.shap._compute_shap_values") as mock_compute:
        # Set up the mock
        mock_compute.return_value = MagicMock(spec=shap.Explanation)
        mock_compute.return_value.values = np.zeros((X.shape[0], X.shape[1]))

        # Call the function
        calculate_shap_explanation(model=tree_model, X=X, check_additivity=check_additivity, random_state=42)

        # Verify the mock was called with the correct check_additivity parameter
        _, call_kwargs = mock_compute.call_args
        assert call_kwargs["check_additivity"] == check_additivity


@pytest.mark.parametrize(
    "data_size, sample_size, expected_sample_size",
    [
        (100, 50, 50),  # Normal case - dataset larger than sample_size
        (30, 100, 6),  # Small dataset case - should use 20% of data
        (200, 0, 0),  # Edge case - sample_size is 0
    ],
)
def test_create_shap_explainer_sample_size(tree_model, data_size, sample_size, expected_sample_size):
    """Test sample_size parameter in _create_shap_explainer."""
    # Create dataset of specified size
    X = pd.DataFrame(np.random.rand(data_size, 5), columns=[f"f{i}" for i in range(5)])

    # Mock both the sample function and the Explainer class
    with patch("probatus.utils.shap.sample") as mock_sample, patch("probatus.utils.shap.Explainer") as mock_explainer:
        # Set up the mocks
        mock_sample.return_value = "mock_masker"
        mock_explainer.return_value = MagicMock()

        # Call the function
        _create_shap_explainer(model=tree_model, X=X, random_state=42, sample_size=sample_size)

        # Verify sample was called with the correct parameters
        if sample_size == 0:
            # Even with sample_size=0, the code will still call sample (with nsamples=0)
            mock_sample.assert_called_once()
            args, kwargs = mock_sample.call_args
            assert kwargs.get("nsamples", args[1] if len(args) > 1 else None) == 0
        else:
            mock_sample.assert_called_once()
            args, kwargs = mock_sample.call_args
            assert kwargs.get("nsamples", args[1] if len(args) > 1 else None) == expected_sample_size

        # Verify Explainer was called
        mock_explainer.assert_called_once()
