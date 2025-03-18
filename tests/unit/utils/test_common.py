import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probatus.utils import handle_class_names, assure_list_of_strings, is_regression_model


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


def test_handle_class_names(dependencies_classification_model, dependencies_classification_data):
    """Test the handle_class_names function with different inputs."""
    X, y = dependencies_classification_data
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
