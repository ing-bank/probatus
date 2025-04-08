import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification

from probatus.wrapper import (
    calculate_shap_explanation,
    calculate_shap_importance_dataframe,
    create_importance_dataframe,
    calculate_base_shap_statistics,
    aggregate_multiclass_shap_values_values,
    extract_shap_parameters,
    process_shap_values,
    shap_explanation_to_shap_values,
)
from probatus.wrapper._shap.values import _get_shap_values_for_class, _apply_class_weighting


@pytest.mark.parametrize(
    "model_fixture, input_type, sample_size, expected_behavior",
    [
        ("tree_model", "dataframe", None, "basic"),
        ("linear_model", "dataframe", None, "basic"),
        ("tree_model", "array_with_categorical", None, "with_categorical"),
        ("tree_model", "small_dataset", None, "small_data"),
        ("tree_model", "small_dataset", 10, "use_sample"),  # Testing sample_size
        ("tree_model", "small_dataset", 100, "use_sample"),  # Large sample_size
        ("tree_model", "small_dataset", 1, "use_sample"),  # Minimum valid sample size
    ],
)
def test_create_shap_explainer_comprehensive(
    request, binary_classification_data, model_fixture, input_type, sample_size, expected_behavior
):
    """Test _create_shap_explainer with different models, data types, and sample sizes."""
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
        input_data = pd.DataFrame(np.random.rand(20, 3), columns=[f"f{i}" for i in range(3)])

    # Test explainer creation with optional sample_size
    kwargs = {}
    if sample_size is not None:
        kwargs["sample_size"] = sample_size

    explainer = _create_shap_explainer(model, input_data, random_state=42, **kwargs)

    # Check the result
    assert isinstance(explainer, shap.Explainer)

    # Additional verification for specific behaviors is difficult
    # since the explainer's internals vary by model type,
    # but we can check that it was created successfully
    assert explainer is not None


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
        (None, None),  # Default behavior - select first class
        ("max_abs", None),  # Max absolute aggregation
        ("mean_abs", None),  # Mean absolute aggregation
        (None, 1),  # Specific class by index
    ],
)
def test_process_shap_values_multiclass(multi_classification_data, multiclass_aggregation, class_selection):
    """Test multiclass aggregation in both _process_shap_values and shap_explanation_to_shap_values."""
    # Use real multiclass data and get SHAP explanation
    X, y = multi_classification_data

    # Create and train a multiclass model
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # 1. Test _process_shap_values function
    formatted = process_shap_values(
        shap_explanation=shap_explanation,
        classes=model.classes_,
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
    elif multiclass_aggregation == "mean_abs":
        # Check that values match mean absolute across classes
        expected = np.mean(np.abs(shap_explanation.values), axis=2)
        np.testing.assert_array_equal(formatted, expected)
    elif class_selection is not None:
        # Check that values match the specified class
        class_idx = list(model.classes_).index(int(class_selection))
        expected = shap_explanation.values[:, :, class_idx]
        np.testing.assert_array_equal(formatted, expected)
    else:
        # Default behavior should select first class
        expected = shap_explanation.values
        np.testing.assert_array_equal(formatted, expected)

    # 2. Test corresponding functionality in shap_explanation_to_shap_values
    # Only run this part for scenarios with multiclass_aggregation but no class_selection
    if class_selection is None:
        # Convert to DataFrame with the same aggregation method
        shap_df = shap_explanation_to_shap_values(
            shap_explanation=shap_explanation, model=model, X=X, multiclass_aggregation=multiclass_aggregation
        )

        if shap_df.ndim == 3:
            assert isinstance(shap_df, np.ndarray)
            assert shap_df.shape == (X.shape[0], X.shape[1], len(np.unique(y)))
        else:
            assert isinstance(shap_df, pd.DataFrame)
            assert shap_df.shape == X.shape
            assert list(shap_df.columns) == list(X.columns)
            assert list(shap_df.index) == list(X.index)


def test_shap_explanation_to_shap_values_basic(tree_model, binary_classification_data):
    """Test basic functionality of shap_explanation_to_shap_values function."""
    X, _ = binary_classification_data

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=tree_model, X=X, return_explainer=False, random_state=42)

    # Convert to DataFrame with default settings
    shap_df = shap_explanation_to_shap_values(shap_explanation=shap_explanation, model=tree_model, X=X)

    # Check output
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)
    assert list(shap_df.index) == list(X.index)


@pytest.mark.parametrize(
    "weights, expected_weight_effect",
    [
        ({0: 0.5, 1: 0.3, 2: 0.2}, "custom_weight"),  # Custom weights by index
        ({0: 1.0, 1: 0.0, 2: 0.0}, "any_weight"),  # Any custom weights should be accepted
    ],
)
def test_process_shap_values_with_weights(multi_classification_data, weights, expected_weight_effect):
    """Test _process_shap_values with different weighting strategies using real data."""
    X, y = multi_classification_data

    # Create and train a multiclass model
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # Get unweighted values (first class) as reference
    unweighted_values = shap_explanation.values[:, :, 0].copy()

    # Test weighted values
    weighted_values = process_shap_values(shap_explanation=shap_explanation, weights=weights)

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


def test_process_shap_values_invalid_weights(multi_classification_data):
    """Test _process_shap_values with an invalid weights."""
    # Create a real multiclass model and SHAP explanation instead of a mock
    X, y = multi_classification_data
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate real SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # Try to format with an invalid weights
    with pytest.raises(ValueError, match="Unsupported weights: invalid_type. Use a dictionary of weights."):
        process_shap_values(shap_explanation, weights="invalid_type", classes=model.classes_)


def test_process_shap_values_invalid_class_selection(multi_classification_data):
    """Test _process_shap_values with an invalid class selection."""
    # Create a real multiclass model and SHAP explanation instead of a mock
    X, y = multi_classification_data
    model = RandomForestClassifier(random_state=42, n_estimators=2)
    model.fit(X, y)

    # Generate real SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # Get the highest class number from the output_names and add 1 to ensure it's invalid
    max_class = max(shap_explanation.output_names)
    invalid_class = max_class + 1 if isinstance(max_class, int) else len(shap_explanation.output_names)

    # Try to format with an invalid class selection
    with pytest.raises(IndexError, match="index 5 is out of bounds for axis 2 with size 5"):
        process_shap_values(shap_explanation, class_selection=invalid_class, classes=model.classes_)


@pytest.mark.parametrize(
    "model_fixture, sample_size, check_additivity, return_explainer",
    [
        ("tree_model", 100, False, False),
        ("linear_model", 50, True, False),
        ("tree_model", 20, False, True),
        ("tree_model", 5, True, False),  # Added case to specifically test check_additivity
    ],
)
def test_calculate_shap_explanation(
    request, binary_classification_data, model_fixture, sample_size, check_additivity, return_explainer
):
    """Test calculate_shap_explanation with different arguments, including check_additivity."""
    model = request.getfixturevalue(model_fixture)
    X, _ = binary_classification_data

    # Use a smaller sample for speed in some test cases
    if sample_size <= 5:
        X = X.iloc[:5]

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

    # Verify that the calculation completes successfully with expected dimensions
    # Note: We can't determine exactly how many samples SHAP will use
    # since it depends on the implementation details of the Explainer
    assert shap_explanation.values.shape[0] > 0  # Should have at least some samples
    assert shap_explanation.values.shape[1] == X.shape[1]  # Should match feature count


def test_shap_values_with_pipeline(pipeline_model, binary_classification_data):
    """Test that SHAP values can be calculated for a pipeline model."""
    X, _ = binary_classification_data

    # This should now work without errors
    shap_explanation = calculate_shap_explanation(pipeline_model, X)

    # Make sure we got a valid SHAP explanation
    assert isinstance(shap_explanation, shap.Explanation)
    assert shap_explanation.values.shape[0] == X.shape[0]  # Should have the right number of samples


@pytest.mark.parametrize(
    "weights, class_selection, data_type, expected_effect",
    [
        # Multiclass data test cases
        (None, None, "multiclass", None),  # Default behavior
        ({0: 0.7, 4: 0.15, 1: 0.05, 2: 0.05, 3: 0.05}, None, "multiclass", "custom"),  # Custom weights
        (None, "0", "multiclass", None),  # Specific class
        # Binary data test cases
        (None, None, "binary", "baseline"),  # Default behavior with binary
        ({1: 2.0, 0: 1.0}, None, "binary", "custom"),  # Custom weights with binary
    ],
)
def test_shap_explanation_to_shap_values_with_weights(
    request, tree_model, weights, class_selection, data_type, expected_effect
):
    """Test shap_explanation_to_shap_values and _shap_values_to_df functions with weighting for all data types."""
    # Get appropriate data based on data_type
    if data_type == "multiclass":
        X, y = request.getfixturevalue("multi_classification_data")
        # Create and train a multiclass model
        model = RandomForestClassifier(random_state=42, n_estimators=2)
        model.fit(X, y)
    else:  # binary
        X, _ = request.getfixturevalue("binary_classification_data")
        model = tree_model  # Use the provided tree_model for binary classification

    # Generate SHAP values
    shap_explanation = calculate_shap_explanation(model=model, X=X, return_explainer=False, random_state=42)

    # For comparison in binary cases or custom weighting tests
    if data_type == "binary" or expected_effect == "custom":
        unweighted_df = shap_explanation_to_shap_values(shap_explanation=shap_explanation, model=model, X=X)

    # Create weighted/class-selected version
    weighted_df = shap_explanation_to_shap_values(
        shap_explanation=shap_explanation, model=model, X=X, weights=weights, class_selection=class_selection
    )

    # Check output structure
    if data_type == "binary":
        assert isinstance(weighted_df, pd.DataFrame)
        assert weighted_df.shape == X.shape
        assert list(weighted_df.columns) == list(X.columns)
        assert list(weighted_df.index) == list(X.index)

    # Check effect of weighting for binary classification
    if expected_effect == "baseline":
        # This is the reference case (no weighting)
        pass
    elif expected_effect == "custom":
        # Any custom weighting should produce valid output
        assert weighted_df is not None
        assert not weighted_df.isnull().any().any(), "Weighted DataFrame should not contain NaNs"

        # For custom weights in multiclass, verify they have an effect compared to unweighted
        if weights is not None and isinstance(weights, dict) and weights != {}:
            # For binary classification, weights are ignored, so we don't expect differences
            if data_type == "binary":
                # Just verify the values are the same as unweighted - weights don't apply to binary classification
                assert np.allclose(
                    weighted_df.values, unweighted_df.values, rtol=1e-3, atol=1e-3
                ), "For binary classification, weights should have no effect"
            else:
                # For multiclass, we do expect differences with custom weights
                is_different = not np.allclose(weighted_df.values, np.sum(unweighted_df, axis=2), rtol=1e-3, atol=1e-3)
                assert is_different, "Custom weights should produce different SHAP values for multiclass"


@pytest.mark.parametrize(
    "input_type, has_precalc, class_selection, multiclass_aggregation",
    [
        ("dataframe", True, None, None),
        ("dataframe", False, None, "max_abs"),
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
        with pytest.raises(ValueError, match="Precalculated SHAP values are empty"):
            _shap_values_to_df(model=tree_model, X=input_data)
    else:
        if has_precalc:
            precalc_shap = np.random.rand(*shape)
            shap_df = _shap_values_to_df(model=tree_model, X=input_data, precalc_shap=precalc_shap)
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
    "data_source, input_type, columns_arg, suffix, variance_penalty",
    [
        # Basic synthetic data cases
        ("synthetic", "numpy", "feature_names", None, None),
        ("synthetic", "dataframe", None, None, None),
        ("synthetic", "numpy", "custom", None, None),
        # Suffix and penalty test cases
        ("synthetic", "dataframe", None, "_test", 0.5),
        ("synthetic", "numpy", "feature_names", "_model1", 0.8),
        ("synthetic", "dataframe", None, "_comparison", -0.1),
        # Real multiclass data cases
        ("multiclass", None, None, None, None),
        ("multiclass", None, None, None, 0.5),
    ],
)
def test_calculate_shap_importance_comprehensive(
    shap_input_data, multi_classification_data, data_source, input_type, columns_arg, suffix, variance_penalty
):
    """Test calculate_shap_importance with both synthetic and real SHAP values, covering all options."""
    kwargs = {}
    if suffix:
        kwargs["output_columns_suffix"] = suffix
    if variance_penalty is not None:
        kwargs["shap_variance_penalty_factor"] = variance_penalty

    if data_source == "synthetic":
        # Use synthetic data from fixture
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

        # Calculate importance
        importance = calculate_shap_importance_dataframe(shap_values, columns=columns, **kwargs)

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

    elif data_source == "multiclass":
        # Use real multiclass data
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
        importance = calculate_shap_importance_dataframe(shap_explanation.values, columns=feature_names, **kwargs)

        # Check results
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == len(feature_names)  # Should have one row per feature
        assert "mean_abs_shap_value" in importance.columns
        assert "mean_shap_value" in importance.columns

        # Check that all feature names are in the index
        for feature in feature_names:
            assert feature in importance.index

        # Check variance penalty if applied
        if variance_penalty is not None and variance_penalty > 0:
            # Verify penalized column exists and has lower values than non-penalized
            assert "penalized_mean_abs_shap_value" in importance.columns
            assert (importance["penalized_mean_abs_shap_value"] <= importance["mean_abs_shap_value"]).all()


def test_calculate_shap_importance_dimension_mismatch(shap_input_data):
    """Test that a dimension mismatch raises an appropriate error."""
    shap_values = shap_input_data["numpy_2d"]
    wrong_columns = [f"feature_{i}" for i in range(shap_values.shape[1] - 1)]  # One fewer column

    with pytest.raises(ValueError, match="Dimension mismatch"):
        calculate_shap_importance_dataframe(shap_values, columns=wrong_columns)


def test_calculate_shap_explanation_with_pipeline():
    """Test that SHAP values can be calculated for a pipeline model with preprocessing."""
    # Create a more complex dataset with numeric and categorical features
    X = pd.DataFrame(
        {
            "num1": np.random.normal(0, 1, 100),
            "num2": np.random.normal(0, 1, 100),
            "cat1": np.random.choice(["A", "B", "C"], 100),
        }
    )
    y = (X["num1"] + np.random.normal(0, 0.1, 100) > 0).astype(int)

    # Create a column transformer for mixed data types
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["num1", "num2"]),
            ("cat", OneHotEncoder(sparse_output=False), ["cat1"]),
        ]
    )

    # Create a pipeline with preprocessing and estimator
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(random_state=42))])

    # Train the pipeline
    pipeline.fit(X, y)

    # Calculate SHAP values with our updated function that handles pipelines
    shap_explanation = calculate_shap_explanation(pipeline, X)

    # Calculate the number of feature columns after preprocessing
    # 2 numeric features + one-hot encoded categorical feature (3 categories)
    expected_feature_count = 2 + 3

    # Verify that SHAP values have the correct shape
    assert shap_explanation.values.shape[0] == 100  # 100 samples
    assert shap_explanation.values.shape[1] == expected_feature_count  # Features after preprocessing

    # Create a non-pipeline model for comparison
    simple_model = LogisticRegression(random_state=42)

    # Manually apply the preprocessing
    X_transformed = pipeline.named_steps["preprocessor"].transform(X)

    # Train the simple model on the preprocessed data
    simple_model.fit(X_transformed, y)

    # Convert X_transformed to DataFrame for SHAP function
    X_transformed_df = pd.DataFrame(X_transformed, columns=[f"feature_{i}" for i in range(X_transformed.shape[1])])

    # Calculate SHAP values for the simple model
    shap_explanation_simple = calculate_shap_explanation(simple_model, X_transformed_df)

    # Both should have the same shape for the SHAP values
    assert shap_explanation.values.shape == shap_explanation_simple.values.shape


@pytest.mark.parametrize(
    "test_case",
    [
        # Test case 1: String class names
        {
            "name": "string_selection",
            "model_classes": ["class_0", "class_1", "class_2"],
            "class_selection": "class_1",
            "expected_index": 1,
            "expected_error": None,
        },
        # Test case 4: Invalid class name
        {
            "name": "invalid_class",
            "model_classes": ["class_0", "class_1", "class_2"],
            "class_selection": "invalid_class",
            "expected_index": None,
            "expected_error": ValueError(
                "Class 'invalid_class' not found in model classes: \\['class_0', 'class_1', 'class_2'\\]"
            ),
        },
        # Test case 6: String class values
        {
            "name": "string_classes_not_sorted",
            "model_classes": ["high", "medium", "low"],
            "class_selection": "medium",
            "expected_index": 2,
            "expected_error": None,
        },
        # Test case 7: Integers as strings
        {
            "name": "string_integer_classes",
            "model_classes": ["1", "2", "3"],
            "class_selection": "2",
            "expected_index": 1,
            "expected_error": None,
        },
        # Test case 8: Integer treated as index
        {
            "name": "integer_as_index",
            "model_classes": [10, 20, 30],
            "class_selection": 1,
            "expected_index": 1,
            "expected_error": None,
        },
        # Test case 9: Invalid index
        {
            "name": "invalid_index",
            "model_classes": [10, 20, 30],
            "class_selection": 5,
            "expected_index": None,
            "expected_error": IndexError("index 5 is out of bounds for axis 2 with size 3"),
        },
    ],
)
def test_get_shap_values_for_class(test_case):
    """Test getting SHAP values for different class selection scenarios."""
    # Create real data and model
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Modify y to match the model_classes we want to test
    # This ensures the model is trained with the correct class labels
    y = np.array([test_case["model_classes"][i] for i in y])

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Create real SHAP explanation using probatus
    shap_explanation = calculate_shap_explanation(model, X, random_state=42)

    # Test the function
    if test_case["expected_error"] is not None:
        with pytest.raises(type(test_case["expected_error"]), match=str(test_case["expected_error"])):
            _get_shap_values_for_class(shap_explanation, test_case["class_selection"], model.classes_)
    else:
        result = _get_shap_values_for_class(shap_explanation, test_case["class_selection"], model.classes_)

        # Check shape and values
        assert result.shape == (100, 5)  # Should be 2D (samples, features)
        np.testing.assert_array_equal(result, shap_explanation.values[:, :, test_case["expected_index"]])


def test_get_shap_values_for_class_empty_classes():
    """Test handling of empty model classes list."""
    # Create real data and model
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Create real SHAP explanation using probatus
    shap_explanation = calculate_shap_explanation(model, X, random_state=42)

    # Test with empty class list
    model_classes = []

    # Should raise ValueError for empty class list
    with pytest.raises(ValueError, match="Class 'class_0' not found in model classes: \\[\\]"):
        _get_shap_values_for_class(shap_explanation, "class_0", model_classes)
