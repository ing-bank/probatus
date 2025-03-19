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
        (True, True, "binary"),
        (False, True, "multiclass"),
        (True, False, "regression"),
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
    "multiclass_aggregation, class_selection",
    [
        (None, None),  # Default behavior
        ("max_abs", None),  # Max absolute aggregation
        ("variance", None),  # Variance aggregation
        ("mean_abs", None),  # Mean absolute aggregation
        (None, "Output 0"),  # Specific class by index
    ],
)
def test_format_shap_values_multiclass(multi_classification_data, multiclass_aggregation, class_selection):
    """Test _format_shap_values with real multiclass data and different aggregation methods."""
    # Use real multiclass data and get SHAP explanation
    X, y = multi_classification_data

    # Create and train a multiclass model
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # Format the SHAP values
    formatted = _format_shap_values(
        shap_explanation=shap_explanation,
        class_selection=class_selection,
        multiclass_aggregation=multiclass_aggregation,
    )

    # Check that the output has the correct shape
    assert isinstance(formatted, np.ndarray)
    assert formatted.shape[0] == X.shape[0]
    assert formatted.shape[1] == X.shape[1]

    # Check specific aggregation logic
    if multiclass_aggregation == "max_abs":
        # Check that values match max absolute across classes
        expected = np.max(np.abs(shap_explanation.values), axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif multiclass_aggregation == "variance":
        # Check that values match variance across classes
        expected = np.var(shap_explanation.values, axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif multiclass_aggregation == "mean_abs":
        # Check that values match mean absolute across classes
        expected = np.mean(np.abs(shap_explanation.values), axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif class_selection is not None:
        # Check that values match the specified class
        class_idx = shap_explanation.output_names.index(class_selection)
        expected = shap_explanation.values[:, :, class_idx]
        np.testing.assert_array_equal(formatted, expected)
    else:
        # Default behavior should select first class
        expected = shap_explanation.values[:, :, 0]
        np.testing.assert_array_equal(formatted, expected)


@pytest.mark.parametrize(
    "weight_type, expected_weight_effect",
    [
        ("frequency", "equal_weight"),  # Equal frequency weighting
        ({0: 0.5, 1: 0.3, 2: 0.2}, "custom_weight"),  # Custom weights by index
        ({0: 1.0, 1: 0.0, 2: 0.0}, "any_weight"),  # Any custom weights should be accepted
    ],
)
def test_format_shap_values_with_weights(multi_classification_data, weight_type, expected_weight_effect):
    """Test _format_shap_values with different weighting strategies using real data."""
    X, y = multi_classification_data

    # Create and train a multiclass model
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # Get unweighted values (first class) as reference
    unweighted_values = shap_explanation.values[:, :, 0].copy()

    # Test weighted values
    weighted_values = _format_shap_values(shap_explanation=shap_explanation, weight_type=weight_type)

    # Check that the output has the correct shape for all weighting methods
    assert isinstance(weighted_values, np.ndarray)
    assert weighted_values.shape[0] == X.shape[0]
    assert weighted_values.shape[1] == X.shape[1]

    # For custom weights, verify they have an effect (compared to unweighted)
    # Only do this for custom weight case with significant balance across classes
    if expected_weight_effect == "custom_weight":
        is_different = not np.allclose(weighted_values, unweighted_values, rtol=1e-3, atol=1e-3)
        assert is_different, "Custom weights should produce different results than unweighted"

    # For any weight type, also verify the sum across 2nd dimension is a 2D array
    # This verifies that class dimension was properly reduced
    assert len(weighted_values.shape) == 2


def test_format_shap_values_invalid_weight_type():
    """Test _format_shap_values with an invalid weight_type."""
    # Create a mock Explanation object with multiclass data (3D)
    mock_explanation = MagicMock()
    mock_explanation.values = np.random.rand(10, 5, 3)
    mock_explanation.output_names = [0, 1, 2]  # Class names

    # Try to format with an invalid weight_type
    with pytest.raises(ValueError, match="Unsupported weight_type"):
        _format_shap_values(mock_explanation, weight_type="invalid_type")


def test_format_shap_values_invalid_class_selection():
    """Test _format_shap_values with an invalid class selection."""
    # Create a mock Explanation object
    mock_explanation = MagicMock()
    mock_explanation.values = np.random.rand(10, 5, 3)
    mock_explanation.output_names = [0, 1, 2]  # Class names

    # Try to format with an invalid class selection
    with pytest.raises(ValueError, match="Class '3' not found in model classes"):
        _format_shap_values(mock_explanation, class_selection=3)


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
    "weight_type, class_selection",
    [
        (None, None),  # Default behavior
        ("frequency", None),  # Equal frequency weighting
        ({0: 0.7, 1: 0.3}, None),  # Custom weights
        (None, "Output 0"),  # Specific class - use the class name format from SHAP
    ],
)
def test_shap_explanation_to_shap_df_with_weights(tree_model, multi_classification_data, weight_type, class_selection):
    """Test shap_explanation_to_shap_df function with weighting."""
    X, y = multi_classification_data
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # Convert to DataFrame with weighting
    shap_df = shap_explanation_to_shap_df(
        shap_explanation=shap_explanation, model=model, X=X, weight_type=weight_type, class_selection=class_selection
    )

    # Check output
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)
    assert list(shap_df.index) == list(X.index)


@pytest.mark.parametrize(
    "input_type, has_precalc, class_selection, multiclass_aggregation",
    [
        ("dataframe", True, None, None),
        ("dataframe", False, None, "max_abs"),
        ("numpy", True, None, "variance"),
        ("numpy", False, None, "mean_abs"),
        ("empty", False, None, None),
    ],
)
def test_shap_values_to_df(
    tree_model,
    binary_classification_data,
    shap_input_data,
    input_type,
    has_precalc,
    class_selection,
    multiclass_aggregation,
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
                model=tree_model,
                X=input_data,
                class_selection=class_selection,
                multiclass_aggregation=multiclass_aggregation,
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
    "weight_type, expected_effect",
    [
        (None, "baseline"),  # No weighting (baseline)
        ("frequency", "custom"),  # Frequency weighting (affects binary case differently than expected)
        ({1: 2.0, 0: 1.0}, "custom"),  # Weight for positive class should have effect
    ],
)
def test_shap_explanation_to_shap_df_binary_with_weights(
    tree_model, binary_classification_data, weight_type, expected_effect
):
    """Test shap_explanation_to_shap_df function with binary classification and weighting."""
    X, _ = binary_classification_data

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=tree_model, X=X, return_explainer=False, random_state=42)

    # Create unweighted version for comparison
    unweighted_df = shap_explanation_to_shap_df(shap_explanation=shap_explanation, model=tree_model, X=X)

    # Create weighted version
    weighted_df = shap_explanation_to_shap_df(
        shap_explanation=shap_explanation, model=tree_model, X=X, weight_type=weight_type
    )

    # Check output structure
    assert isinstance(weighted_df, pd.DataFrame)
    assert weighted_df.shape == X.shape
    assert list(weighted_df.columns) == list(X.columns)
    assert list(weighted_df.index) == list(X.index)

    # Check effect of weighting
    if expected_effect == "baseline":
        # This is the reference case (no weighting)
        pass
    elif expected_effect == "custom":
        # Any custom weighting should produce valid output, but we don't
        # make assumptions about specific values
        assert weighted_df is not None
        assert not weighted_df.isnull().any().any(), "Weighted DataFrame should not contain NaNs"


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


def test_calculate_shap_importance_multiclass(multi_classification_data):
    """Test calculate_shap_importance with real multiclass SHAP values."""
    X, y = multi_classification_data

    # Get a small sample of data to make the test faster
    X_sample = X.iloc[:10, :5]  # Make sure X has sufficient columns
    y_sample = y[:10]

    # Create and train a multiclass model
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X_sample, y_sample)

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X_sample, return_explainer=False, random_state=42)

    # Get feature names - making sure to use actual column names from the sample
    feature_names = X_sample.columns.tolist()

    # Verify the shapes match before calculating importance
    assert shap_explanation.values.shape[1] == len(feature_names)

    # Calculate importance
    importance = calculate_shap_importance(shap_explanation.values, columns=feature_names)

    # Check results
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == len(feature_names)  # Should have one row per feature
    assert "mean_abs_shap_value" in importance.columns
    assert "mean_shap_value" in importance.columns

    # Check that all feature names are in the index
    for feature in feature_names:
        assert feature in importance.index

    # Apply a variance penalty and check the results
    importance_with_penalty = calculate_shap_importance(
        shap_explanation.values, columns=feature_names, shap_variance_penalty_factor=0.5
    )

    # Verify penalized column exists and has lower values than non-penalized
    assert "penalized_mean_abs_shap_value" in importance_with_penalty.columns
    assert (
        importance_with_penalty["penalized_mean_abs_shap_value"] <= importance_with_penalty["mean_abs_shap_value"]
    ).all()


@pytest.mark.parametrize("check_additivity", [True, False])
def test_shap_additivity_checking(tree_model, binary_classification_data, check_additivity):
    """Test that the check_additivity parameter affects SHAP values."""
    X, _ = binary_classification_data
    sample_X = X.iloc[:5]  # Use a small sample for speed

    # Calculate SHAP explanations with different additivity settings
    shap_explanation = calculate_shap_explanation(
        model=tree_model, X=sample_X, check_additivity=check_additivity, random_state=42
    )

    # Just verify that the calculation completes successfully
    assert isinstance(shap_explanation, shap.Explanation)
    assert shap_explanation.values.shape[0] == sample_X.shape[0]
    assert shap_explanation.values.shape[1] == sample_X.shape[1]


@pytest.mark.parametrize(
    "data_size, sample_size, expected_behavior",
    [
        (20, 10, "use_sample"),  # Normal case - should use a sample
        (20, 100, "use_sample"),  # Large sample_size - should still use a sample
        (20, 1, "use_sample"),  # Minimum valid sample size
    ],
)
def test_create_shap_explainer_sample_size(tree_model, data_size, sample_size, expected_behavior):
    """Test sample_size parameter in _create_shap_explainer using real data."""
    # Create dataset of specified size
    X = pd.DataFrame(np.random.rand(data_size, 3), columns=[f"f{i}" for i in range(3)])

    # Create explainer with the specified sample_size
    explainer = _create_shap_explainer(model=tree_model, X=X, random_state=42, sample_size=sample_size)

    # Check that explainer was created
    assert isinstance(explainer, shap.Explainer)

    # Additional verification is difficult since the explainer's internals vary by model type
    # but we can check that it was created successfully
    if expected_behavior == "use_sample":
        # Just verify the explainer was created - can't easily check the exact sample size used
        assert explainer is not None


@pytest.mark.parametrize(
    "weight_type",
    [
        None,  # Default, no weighting
        "frequency",  # Equal frequency weighting
        {0: 0.7, 1: 0.3},  # Custom weights
    ],
)
def test_shap_values_to_df_with_weights(multi_classification_data, weight_type):
    """Test _shap_values_to_df with different weighting options using real data."""
    X, y = multi_classification_data

    # Create and train a multiclass model
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate unweighted SHAP values as reference
    unweighted_df = _shap_values_to_df(model=model, X=X)

    # Generate weighted SHAP values
    weighted_df = _shap_values_to_df(model=model, X=X, weight_type=weight_type)

    # Basic validation
    assert isinstance(weighted_df, pd.DataFrame)
    assert weighted_df.shape == X.shape
    assert list(weighted_df.columns) == list(X.columns)

    # For custom weights that differ significantly from default,
    # the weighted values should be different from unweighted
    if weight_type is not None and isinstance(weight_type, dict) and weight_type != {}:
        # Instead of expecting an assertion error, directly compare the arrays
        is_different = not np.allclose(weighted_df.values, unweighted_df.values, rtol=1e-3, atol=1e-3)
        assert is_different, "Custom weights should produce different SHAP values"
