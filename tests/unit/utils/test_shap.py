import numpy as np
import pandas as pd
import pytest
import shap

from probatus.utils.shap import (
    _validate_shap_inputs,
    _create_shap_explainer,
    _compute_shap_values,
    _format_shap_values,
    shap_calc,
    shap_to_df,
    calculate_shap_importance,
)


@pytest.mark.parametrize(
    "model_fixture, expected_valid, expected_error_contains",
    [
        ("tree_model", True, None),
        ("linear_model", True, None),
        ("pipeline_model", False, "Pipeline"),
    ],
)
def test_validate_shap_inputs(request, classification_data, model_fixture, expected_valid, expected_error_contains):
    """Test _validate_shap_inputs with various model types."""
    model = request.getfixturevalue(model_fixture)
    X, _ = classification_data

    # Test validation
    is_valid, error_message = _validate_shap_inputs(model, X, verbose=0)
    assert is_valid is expected_valid

    if expected_error_contains:
        assert expected_error_contains in error_message


def test_validate_shap_inputs_non_dataframe(tree_model, classification_data):
    """Test _validate_shap_inputs with non-DataFrame input."""
    X, _ = classification_data
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
def test_create_shap_explainer(request, classification_data, model_fixture, input_type):
    """Test _create_shap_explainer with different models and data types."""
    model = request.getfixturevalue(model_fixture)
    X, _ = classification_data

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


@pytest.mark.parametrize("approximate", [True, False])
def test_compute_shap_values(shap_explainer, classification_data, approximate):
    """Test computing SHAP values with different approximation settings."""
    X, _ = classification_data

    # Compute SHAP values
    shap_values = _compute_shap_values(shap_explainer, X, approximate=approximate)

    # Check shape consistency
    if isinstance(shap_values, list):
        assert len(shap_values) == 2  # Binary classification
        assert shap_values[0].shape == (X.shape[0], X.shape[1])
    else:
        assert shap_values.shape[0] == X.shape[0]
        assert shap_values.shape[1] == X.shape[1]


@pytest.mark.parametrize(
    "input_type, expected_shape, expected_warning",
    [
        ("list_binary", "2d", None),
        ("numpy_3d_binary", "2d", None),
        ("numpy_3d_single", "2d", "Could not extract dimension 1"),
        ("list_single", "2d", "Converting list of SHAP values"),
    ],
)
def test_format_shap_values(shap_input_data, input_type, expected_shape, expected_warning):
    """Test formatting different SHAP value structures."""
    X_shape = shap_input_data["X_shape"]
    input_data = shap_input_data[input_type]

    if expected_warning:
        with pytest.warns(UserWarning, match=expected_warning):
            formatted = _format_shap_values(input_data, verbose=1)
    else:
        formatted = _format_shap_values(input_data, verbose=0)

    assert isinstance(formatted, np.ndarray)
    if expected_shape == "2d":
        assert formatted.shape == X_shape

    # Validate specific conversion logic
    if input_type == "list_binary":
        np.testing.assert_array_equal(formatted, input_data[1])  # Should select class 1
    elif input_type == "numpy_3d_binary":
        np.testing.assert_array_equal(formatted, input_data[:, :, 1])


@pytest.mark.parametrize(
    "model_fixture, return_explainer",
    [
        ("tree_model", False),
        ("linear_model", False),
        ("tree_model", True),
    ],
)
def test_shap_calc(request, classification_data, model_fixture, return_explainer):
    """Test shap_calc with different models and options."""
    model = request.getfixturevalue(model_fixture)
    X, _ = classification_data

    if return_explainer:
        shap_values, explainer = shap_calc(model, X, return_explainer=True, random_state=42)
        assert isinstance(shap_values, pd.DataFrame)
        assert isinstance(explainer, shap.Explainer)
    else:
        shap_values = shap_calc(model, X, random_state=42)
        assert isinstance(shap_values, pd.DataFrame)
        assert shap_values.shape == X.shape
        assert list(shap_values.columns) == list(X.columns)


def test_shap_calc_with_pipeline(pipeline_model, classification_data):
    """Test that using a Pipeline raises an error."""
    X, _ = classification_data

    with pytest.raises(TypeError, match="Pipeline"):
        shap_calc(pipeline_model, X)


@pytest.mark.parametrize(
    "input_type, has_precalc",
    [
        ("dataframe", True),
        ("dataframe", False),
        ("numpy", True),
        ("empty", False),
    ],
)
def test_shap_to_df(tree_model, classification_data, shap_input_data, input_type, has_precalc):
    """Test shap_to_df with different input types and precalculated values."""
    X, _ = classification_data

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
            shap_to_df(tree_model, input_data)
    else:
        if has_precalc:
            precalc_shap = np.random.rand(*shape)
            shap_df = shap_to_df(tree_model, input_data, precalc_shap)
        else:
            shap_df = shap_to_df(tree_model, input_data)

        assert isinstance(shap_df, pd.DataFrame)
        assert shap_df.shape == shape

        if input_type == "numpy":
            assert shap_df.columns[0] == "col_0"  # Generic column names for arrays


@pytest.mark.parametrize(
    "input_type, columns_arg, suffix, variance_penalty",
    [
        ("numpy", "feature_names", None, None),
        ("dataframe", None, None, None),
        ("numpy", "custom", None, None),
        ("dataframe", None, "_test", 0.5),
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
    if variance_penalty:
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

    if variance_penalty:
        penalized_column = f"penalized_mean_abs_shap_value{suffix or ''}"
        abs_shap_column = f"mean_abs_shap_value{suffix or ''}"
        assert penalized_column in importance.columns
        assert (importance[penalized_column] <= importance[abs_shap_column]).all()


def test_calculate_shap_importance_dimension_mismatch(shap_input_data):
    """Test that a dimension mismatch raises an appropriate error."""
    shap_values = shap_input_data["numpy_2d"]
    wrong_columns = [f"feature_{i}" for i in range(shap_values.shape[1] - 1)]  # One fewer column

    with pytest.raises(ValueError, match="Dimension mismatch"):
        calculate_shap_importance(shap_values, columns=wrong_columns)
