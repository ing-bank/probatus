import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probatus.utils import (
    handle_class_names,
    assure_list_of_strings,
    is_regression_model,
    get_pipeline_estimator_and_preprocessor,
)


def test_assure_list_of_strings_with_string():
    """Test that a single string is converted to a list with one element."""
    result = assure_list_of_strings("test", "variable")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "test"


def test_assure_list_of_strings_with_list_of_strings():
    """Test that a list of strings is returned unchanged."""
    input_list = ["test1", "test2", "test3"]
    result = assure_list_of_strings(input_list, "variable")
    assert result == input_list
    assert result is input_list  # Check that it's the same object


def test_assure_list_of_strings_with_invalid_type():
    """Test that an error is raised for invalid types."""
    with pytest.raises(ValueError) as excinfo:
        assure_list_of_strings(123, "variable_name")
    assert "variable_name needs to be either a string or list of strings" in str(excinfo.value)

    with pytest.raises(ValueError):
        assure_list_of_strings(None, "variable_name")

    with pytest.raises(ValueError):
        assure_list_of_strings({"key": "value"}, "variable_name")


def test_is_regression_model_with_regression_model():
    """Test with a standard regression model."""
    model = LinearRegression()
    assert is_regression_model(model) is True


def test_is_regression_model_with_classification_model():
    """Test with a standard classification model."""
    model = LogisticRegression()
    assert is_regression_model(model) is False


def test_is_regression_model_with_pipeline_regression():
    """Test with a pipeline containing a regression model."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    assert is_regression_model(pipeline) is True


def test_is_regression_model_with_pipeline_classification():
    """Test with a pipeline containing a classification model."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    assert is_regression_model(pipeline) is False


def test_is_regression_model_with_custom_regression_model():
    """Test with a custom model that has predict but not predict_proba."""

    class CustomRegressor:
        def predict(self, X):
            return np.zeros(len(X))

    model = CustomRegressor()
    assert is_regression_model(model) is True


def test_is_regression_model_with_custom_classification_model():
    """Test with a custom model that has both predict and predict_proba."""

    class CustomClassifier:
        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.zeros((len(X), 2))

    model = CustomClassifier()
    assert is_regression_model(model) is False


def test_handle_class_names(dependencies_classification_model, dependencies_binary_classification_data):
    """Test the handle_class_names function with different inputs."""
    X, y = dependencies_binary_classification_data
    is_regression = False
    unique_values = sorted(y.unique())

    # Test with mismatched list length
    with pytest.raises(ValueError, match="Number of class names .* must match number of unique target values"):
        handle_class_names(y, ["Single Class"], is_regression)

    # Test with missing key in dictionary
    incomplete_dict = {unique_values[0]: "Only First Class"}
    with pytest.raises(ValueError, match="Target value .* not found in the class_names dictionary"):
        handle_class_names(y, incomplete_dict, is_regression)

    # Test with invalid class_names type
    with pytest.raises(TypeError, match="class_names must be None, a list of strings, or a dictionary"):
        handle_class_names(y, 123, is_regression)

    # Test with regression model
    is_regression = True
    class_names = handle_class_names(y, None, is_regression)
    assert class_names == ["Regression Output"]

    # Regression model should use provided class names
    class_names = handle_class_names(y, ["Custom Regression"], is_regression)
    assert class_names == ["Custom Regression"]


def test_get_pipeline_preprocessor_and_estimator():
    """Test that Pipeline can be correctly split into preprocessor and estimator."""
    # Create a simple pipeline with preprocessing and estimator steps
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])

    # Test with pipeline
    preprocessor, estimator = get_pipeline_estimator_and_preprocessor(pipeline)

    # Verify preprocessor has the correct steps
    assert isinstance(preprocessor, Pipeline)
    assert len(preprocessor.steps) == 1
    assert preprocessor.steps[0][0] == "scaler"
    assert isinstance(preprocessor.steps[0][1], StandardScaler)

    # Verify estimator is correctly extracted
    assert isinstance(estimator, LogisticRegression)

    # Test with plain estimator
    model = LogisticRegression()
    preprocessor, estimator = get_pipeline_estimator_and_preprocessor(model)

    # Verify preprocessor is None for plain estimator
    assert preprocessor is None

    # Verify estimator is the original model
    assert estimator is model

    # Test with single-step pipeline
    single_step_pipeline = Pipeline([("model", LogisticRegression())])
    preprocessor, estimator = get_pipeline_estimator_and_preprocessor(single_step_pipeline)

    # Verify preprocessor is None for single-step pipeline
    assert preprocessor is None

    # Verify estimator is correctly extracted
    assert isinstance(estimator, LogisticRegression)


def test_preprocess_using_pipeline():
    """Test preprocessing data with pipeline preprocessor."""
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from probatus.utils import preprocess_using_pipeline

    # Create a simple dataset with numeric and categorical features
    X = pd.DataFrame(
        {"num1": [1.0, 2.0, 3.0, 4.0, 5.0], "num2": [0.1, 0.2, 0.3, 0.4, 0.5], "cat1": ["A", "B", "A", "C", "B"]}
    )
    y = np.array([0, 1, 0, 1, 1])

    # Create a column transformer for mixed data types
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["num1", "num2"]),
            ("cat", OneHotEncoder(sparse_output=False), ["cat1"]),
        ]
    )

    # Create a pipeline with preprocessing and estimator
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression())])

    # Fit the pipeline
    pipeline.fit(X, y)

    # Test preprocessing with pipeline
    X_transformed = preprocess_using_pipeline(X, pipeline)

    # Check that X was transformed
    assert isinstance(X_transformed, pd.DataFrame)
    # The output should have 5 columns: 2 numeric features + 3 one-hot encoded categories
    assert X_transformed.shape[1] == 5

    # Test with non-pipeline model
    # Create a numeric-only dataset for the direct model
    X_numeric = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0], "num2": [0.1, 0.2, 0.3, 0.4, 0.5]})
    model = LogisticRegression().fit(X_numeric, y)
    X_unchanged = preprocess_using_pipeline(X_numeric, model)

    # Check that X is unchanged
    assert X_unchanged is X_numeric
