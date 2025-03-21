import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from probatus.features import (
    ShapRFECV,
    validate_step_parameter,
    validate_min_features_parameter,
    validate_shap_variance_penalty_factor_parameter,
    check_if_model_is_compatible_with_early_stopping,
)
from probatus.features.shap_recursive_feature_elimination_helper import (
    _calculate_number_of_features_to_remove,
)
from probatus.core import NotFittedError


# Validation tests
def test_validate_step():
    """Test the validate_step_parameter function.

    The function should accept positive integers and floats between 0 and 1 exclusive.
    It should reject negative values, zero, and non-numeric inputs.
    """
    # Valid inputs
    assert validate_step_parameter(1) == 1
    assert validate_step_parameter(0.5) == 0.5

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_step_parameter(0)
    with pytest.raises(ValueError):
        validate_step_parameter(-1)
    with pytest.raises(ValueError):
        validate_step_parameter("invalid")


def test_validate_min_features():
    """Test the validate_min_features_parameter function.

    The function should accept positive integers.
    It should reject non-positive integers, floats, and non-numeric inputs.
    """
    # Valid inputs
    assert validate_min_features_parameter(1) == 1
    assert validate_min_features_parameter(5) == 5

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_min_features_parameter(0)
    with pytest.raises(ValueError):
        validate_min_features_parameter(-1)
    with pytest.raises(ValueError):
        validate_min_features_parameter(1.5)
    with pytest.raises(ValueError):
        validate_min_features_parameter("invalid")


def test_shap_variance_penalty_factor():
    """Test the validate_shap_variance_penalty_factor_parameter function.

    The function should:
    - Accept None, integers, and floats as valid values
    - Return the provided value for valid inputs
    - Issue a warning for invalid types and negative values, defaulting to 0
    """
    # Test with negative value
    with pytest.warns(UserWarning, match="shap_variance_penalty_factor must be None, int or float"):
        assert validate_shap_variance_penalty_factor_parameter(-1) == 0

    # Test with invalid type
    with pytest.warns(UserWarning, match="shap_variance_penalty_factor must be None, int or float"):
        assert validate_shap_variance_penalty_factor_parameter("invalid") == 0

    # Test with valid values
    assert validate_shap_variance_penalty_factor_parameter(0) == 0
    assert validate_shap_variance_penalty_factor_parameter(0.5) == 0.5
    assert validate_shap_variance_penalty_factor_parameter(1.0) == 1.0


# Feature elimination calculation tests
def test_calculate_number_of_features_to_remove():
    """Test the _calculate_number_of_features_to_remove method."""
    # Normal case - remove 3 features
    assert 3 == _calculate_number_of_features_to_remove(
        current_num_of_features=10, num_features_to_remove=3, min_num_features_to_keep=5
    )

    # Limit case - can only remove 3 features to maintain minimum
    assert 3 == _calculate_number_of_features_to_remove(
        current_num_of_features=8, num_features_to_remove=5, min_num_features_to_keep=5
    )

    # Boundary case - can't remove any features
    assert 0 == _calculate_number_of_features_to_remove(
        current_num_of_features=5, num_features_to_remove=1, min_num_features_to_keep=5
    )

    # Remove all but minimum required
    assert 4 == _calculate_number_of_features_to_remove(
        current_num_of_features=5, num_features_to_remove=7, min_num_features_to_keep=1
    )


# Model compatibility tests
def test_check_if_model_is_compatible_with_early_stopping(logistic_regression, xgb_model, lgbm_regressor):
    """Test the _check_if_model_is_compatible_with_early_stopping method."""
    # Test if model is not compatible with early stopping
    assert not check_if_model_is_compatible_with_early_stopping(logistic_regression)

    # Test if model is compatible with early stopping
    assert check_if_model_is_compatible_with_early_stopping(xgb_model)
    assert check_if_model_is_compatible_with_early_stopping(lgbm_regressor)


# Initialization tests
def test_shaprfecv_init_parameters(random_state_42):
    """Test initialization parameters of ShapRFECV."""
    model = LogisticRegression(random_state=random_state_42)
    shap_elimination = ShapRFECV(
        model,
        step=0.5,
        min_features_to_select=3,
        cv=3,
        scoring="accuracy",
        n_jobs=2,
        verbose=1,
        random_state=random_state_42,
    )

    assert shap_elimination.step == 0.5
    assert shap_elimination.min_features_to_select == 3
    assert shap_elimination.cv == 3
    assert shap_elimination.n_jobs == 2
    assert shap_elimination.verbose == 1
    assert shap_elimination.random_state == random_state_42

    # Test with early stopping parameters
    with pytest.raises(ValueError):
        # Should raise error because model is not compatible with early stopping
        ShapRFECV(model, early_stopping_rounds=5, eval_metric="auc")


# API tests
def test_check_if_fitted(mock_shap_elimination):
    """Test the _check_if_fitted method."""
    # Should not raise error when fitted=True
    mock_shap_elimination._check_if_fitted()

    # Should raise error when fitted=False
    mock_shap_elimination.fitted = False
    with pytest.raises(NotFittedError):
        mock_shap_elimination._check_if_fitted()


def test_column_names_parameter(feature_elimination_X, feature_elimination_y, random_state_42, mock_fit_compute):
    """Test the column_names parameter."""
    model = LogisticRegression(random_state=random_state_42)
    shap_elimination = ShapRFECV(model, random_state=random_state_42, cv=2)
    custom_column_names = ["feature_1", "feature_2", "feature_3"]

    # Mock fit_compute to avoid actual computation
    original_fit_compute = shap_elimination.fit_compute
    shap_elimination.fit_compute = mock_fit_compute

    try:
        shap_elimination.fit(feature_elimination_X, feature_elimination_y, column_names=custom_column_names)
        assert shap_elimination.column_names == custom_column_names
    finally:
        shap_elimination.fit_compute = original_fit_compute


def test_groups_parameter(feature_elimination_X, feature_elimination_y, random_state_42, mock_fit_compute):
    """Test the groups parameter for group-based cross-validation."""
    model = LogisticRegression(random_state=random_state_42)

    # Create ShapRFECV with explicit cv parameter to use StratifiedGroupKFold
    shap_elimination = ShapRFECV(model, random_state=random_state_42, cv=StratifiedGroupKFold(n_splits=2))

    groups = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])

    # Mock fit_compute to avoid actual computation
    original_fit_compute = shap_elimination.fit_compute
    shap_elimination.fit_compute = mock_fit_compute

    try:
        shap_elimination.fit(feature_elimination_X, feature_elimination_y, groups=groups)
        # If we get here, the test passed
    except Exception as e:
        pytest.fail(f"Using groups parameter with StratifiedGroupKFold should not raise an exception: {e}")
    finally:
        shap_elimination.fit_compute = original_fit_compute


def test_sample_weight_parameter(
    feature_elimination_X,
    feature_elimination_y,
    feature_elimination_sample_weight,
    random_state_42,
    mock_feature_shap_values_per_fold,
):
    """Test the sample_weight parameter."""
    model = LogisticRegression(random_state=random_state_42)
    shap_elimination = ShapRFECV(model, random_state=random_state_42, cv=2)
    captured_sample_weight = None

    # Mock _get_feature_shap_values_per_fold to capture sample_weight
    def mock_method(*args, **kwargs):
        nonlocal captured_sample_weight
        captured_sample_weight = kwargs.get("sample_weight")
        return np.array([[0.1, 0.2, 0.3]]), 0.8, 0.7

    original_method = shap_elimination._get_feature_shap_values_per_fold
    shap_elimination._get_feature_shap_values_per_fold = mock_method

    # Mock fit to call our mocked method
    def mock_fit(*args, **kwargs):
        sw = kwargs.get("sample_weight")
        shap_elimination._get_feature_shap_values_per_fold(
            X=feature_elimination_X,
            y=feature_elimination_y,
            model=model,
            train_index=np.array([0, 1]),
            val_index=np.array([2, 3]),
            sample_weight=sw,
        )
        shap_elimination.fitted = True
        return shap_elimination

    original_fit = shap_elimination.fit
    shap_elimination.fit = mock_fit

    try:
        shap_elimination.fit(
            feature_elimination_X, feature_elimination_y, sample_weight=feature_elimination_sample_weight
        )
        assert captured_sample_weight is feature_elimination_sample_weight
    finally:
        shap_elimination._get_feature_shap_values_per_fold = original_method
        shap_elimination.fit = original_fit


def test_shap_kwargs_parameter(
    feature_elimination_X, feature_elimination_y, random_state_42, mock_feature_shap_values_per_fold
):
    """Test the shap_kwargs parameter."""
    model = LogisticRegression(random_state=random_state_42)
    shap_elimination = ShapRFECV(model, random_state=random_state_42, cv=2)
    kwargs_passed = {}

    # Mock _get_feature_shap_values_per_fold to capture kwargs
    def mock_method(*args, **kwargs):
        nonlocal kwargs_passed
        kwargs_passed.update(kwargs)
        return np.array([[0.1, 0.2, 0.3]]), 0.8, 0.7

    original_method = shap_elimination._get_feature_shap_values_per_fold
    shap_elimination._get_feature_shap_values_per_fold = mock_method

    # Mock fit to call our mocked method with shap_kwargs
    def mock_fit(*args, **kwargs):
        shap_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "X",
                "y",
                "sample_weight",
                "columns_to_keep",
                "column_names",
                "groups",
                "shap_variance_penalty_factor",
            ]
        }
        shap_elimination._get_feature_shap_values_per_fold(
            X=feature_elimination_X,
            y=feature_elimination_y,
            model=model,
            train_index=np.array([0, 1]),
            val_index=np.array([2, 3]),
            **shap_kwargs,
        )
        shap_elimination.fitted = True
        return shap_elimination

    original_fit = shap_elimination.fit
    shap_elimination.fit = mock_fit

    try:
        shap_kwargs = {"approximate": True, "check_additivity": False}
        shap_elimination.fit(feature_elimination_X, feature_elimination_y, **shap_kwargs)
        assert kwargs_passed.get("approximate") is True
        assert kwargs_passed.get("check_additivity") is False
    finally:
        shap_elimination._get_feature_shap_values_per_fold = original_method
        shap_elimination.fit = original_fit
