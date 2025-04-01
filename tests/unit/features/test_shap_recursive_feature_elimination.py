import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from probatus.features._shap._importance import (
    _filter_and_identify_features_based_on_importance,
)
from probatus.features._reporting._results import (
    _get_best_num_features,
    _get_feature_names,
    _get_feature_ranking,
    _get_feature_support,
    _report_current_results,
)
from probatus.features._validation._parameters import _validate_model_compatibility_with_early_stopping_parameter
from probatus.features._validation._parameters import (
    _validate_min_features_parameter,
    _validate_shap_variance_penalty_factor_parameter,
    _validate_step_parameter,
)
from probatus.features._shap._importance import (
    _calculate_number_of_features_to_remove,
    _get_current_features_to_remove,
)


@pytest.mark.parametrize(
    "step, expected",
    [
        (1, 1),  # Integer
        (3, 3),  # Integer
        (0.1, 0.1),  # Float
        (0.5, 0.5),  # Float
    ],
)
def test_validate_step_parameter(step, expected):
    """Test step parameter validation."""
    result = _validate_step_parameter(step)
    assert result == expected


@pytest.mark.parametrize(
    "step, expected_error",
    [
        (0, ValueError),  # Zero
        (-1, ValueError),  # Negative
        ("invalid", TypeError),  # Invalid type
    ],
)
def test_validate_step_parameter_errors(step, expected_error):
    """Test step parameter validation errors."""
    with pytest.raises(expected_error):
        _validate_step_parameter(step)


@pytest.mark.parametrize(
    "min_features, expected",
    [
        (1, 1),  # Minimum
        (10, 10),  # Normal
    ],
)
def test_validate_min_features_parameter(min_features, expected):
    """Test min_features_to_select parameter validation."""
    result = _validate_min_features_parameter(min_features)
    assert result == expected


@pytest.mark.parametrize(
    "min_features, expected_error",
    [
        (0, ValueError),  # Zero
        (-1, ValueError),  # Negative
        (1.5, TypeError),  # Float
        ("invalid", TypeError),  # Invalid type
    ],
)
def test_validate_min_features_parameter_errors(min_features, expected_error):
    """Test min_features_to_select parameter validation errors."""
    with pytest.raises(expected_error):
        _validate_min_features_parameter(min_features)


@pytest.mark.parametrize(
    "model_class, params, expected",
    [
        (LGBMClassifier, {"n_estimators": 10, "verbose": -1}, True),
        (XGBClassifier, {"n_estimators": 10}, True),
        (RandomForestClassifier, {"n_estimators": 10}, False),
    ],
)
def test_check_if_model_is_compatible_with_early_stopping(model_class, params, expected, random_state):
    """Test if model compatibility with early stopping is correctly determined."""
    model = model_class(random_state=random_state, **params)
    result = _validate_model_compatibility_with_early_stopping_parameter(model)
    assert result == expected


def test_filter_and_identify_features_based_on_importance():
    """Test filtering features based on SHAP importance."""
    # Create a mock SHAP importance DataFrame
    shap_importance_df = pd.DataFrame(
        {"importance": [0.5, 0.3, 0.2, 0.1]},
        index=["feature1", "feature2", "feature3", "feature4"],
    )
    current_features_set = ["feature1", "feature2", "feature3", "feature4"]

    # Test with integer step
    remaining, removed = _filter_and_identify_features_based_on_importance(
        shap_importance_df,
        step=1,
        min_features_to_select=1,
        columns_to_keep=None,
        current_features_set=current_features_set,
    )
    assert remaining == ["feature1", "feature2", "feature3"]
    assert removed == ["feature4"]

    # Test with float step
    remaining, removed = _filter_and_identify_features_based_on_importance(
        shap_importance_df,
        step=0.5,
        min_features_to_select=1,
        columns_to_keep=None,
        current_features_set=current_features_set,
    )
    assert remaining == ["feature1", "feature2"]
    assert removed == ["feature4", "feature3"]

    # Test with columns_to_keep
    remaining, removed = _filter_and_identify_features_based_on_importance(
        shap_importance_df,
        step=1,
        min_features_to_select=1,
        columns_to_keep=["feature3"],
        current_features_set=current_features_set,
    )
    assert remaining == ["feature1", "feature2", "feature3"]
    assert removed == ["feature4"]


def test_report_current_results():
    """Test creating and updating the report DataFrame."""
    # Initial empty DataFrame
    report_df = pd.DataFrame()

    # First round
    updated_df = _report_current_results(
        report_df=report_df,
        round_number=1,
        current_features_set=["feature1", "feature2", "feature3"],
        features_to_remove=["feature3"],
        train_metric_mean=0.8,
        train_metric_std=0.05,
        val_metric_mean=0.75,
        val_metric_std=0.06,
    )

    assert updated_df.shape[0] == 1
    assert updated_df.loc[1, "num_features"] == 3
    assert updated_df.loc[1, "features_set"] == ["feature1", "feature2", "feature3"]
    assert updated_df.loc[1, "eliminated_features"] == ["feature3"]
    assert updated_df.loc[1, "train_metric_mean"] == 0.8

    # Second round
    updated_df = _report_current_results(
        report_df=updated_df,
        round_number=2,
        current_features_set=["feature1", "feature2"],
        features_to_remove=["feature2"],
        train_metric_mean=0.79,
        train_metric_std=0.04,
        val_metric_mean=0.74,
        val_metric_std=0.05,
    )

    assert updated_df.shape[0] == 2
    assert updated_df.loc[2, "num_features"] == 2
    assert updated_df.loc[2, "features_set"] == ["feature1", "feature2"]
    assert updated_df.loc[2, "eliminated_features"] == ["feature2"]


def test_get_feature_names():
    """Test retrieving feature names for a specific number of features."""
    # Create a mock report DataFrame
    report_df = pd.DataFrame(
        {
            "num_features": [4, 3, 2, 1],
            "features_set": [["f1", "f2", "f3", "f4"], ["f1", "f2", "f3"], ["f1", "f2"], ["f1"]],
        },
        index=[1, 2, 3, 4],
    )

    # Test retrieving feature sets
    assert _get_feature_names(report_df, 4) == ["f1", "f2", "f3", "f4"]
    assert _get_feature_names(report_df, 2) == ["f1", "f2"]

    # Test error case
    with pytest.raises(ValueError):
        _get_feature_names(report_df, 5, True)


def test_get_feature_support():
    """Test generating boolean mask for selected features."""
    column_names = ["f1", "f2", "f3", "f4", "f5"]
    feature_names_selected = ["f1", "f3", "f5"]

    result = _get_feature_support(column_names, feature_names_selected)
    assert result == [True, False, True, False, True]

    # Test error case
    with pytest.raises(ValueError):
        _get_feature_support(None, feature_names_selected)


def test_get_feature_ranking():
    """Test generating feature ranking based on elimination order."""
    # Create a mock report DataFrame
    report_df = pd.DataFrame(
        {
            "features_set": [
                ["f0", "f1", "f3", "f5", "f6", "f7"],
                ["f0", "f1", "f3", "f7"],
                ["f1", "f7"],
            ],
            "eliminated_features": [["f2", "f4"], ["f6", "f5"], ["f3", "f0"]],
        },
        index=[1, 2, 3],
    )

    column_names = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]

    result = _get_feature_ranking(report_df, column_names)
    assert result == [2, 1, 7, 3, 6, 4, 5, 1]

    # Test error case
    with pytest.raises(ValueError):
        _get_feature_ranking(report_df, None)


@pytest.mark.parametrize(
    "penalty_factor, expected",
    [
        (0, 0.0),  # Integer zero
        (1, 1.0),  # Integer
        (0.5, 0.5),  # Float
        (None, 0.0),  # None should default to 0.0
    ],
)
def test_validate_shap_variance_penalty_factor_parameter(penalty_factor, expected):
    """Test validation of shap_variance_penalty_factor parameter."""
    result = _validate_shap_variance_penalty_factor_parameter(penalty_factor)
    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "penalty_factor, expected",
    [
        (-1, 0.0),  # Negative values should issue a warning and return 0.0
        ("invalid", 0.0),  # Invalid types should issue a warning and return 0.0
    ],
)
def test_validate_shap_variance_penalty_factor_parameter_warnings(penalty_factor, expected):
    """Test validation of shap_variance_penalty_factor parameter with invalid values."""
    # These should trigger warnings but still return a default value
    with pytest.warns(UserWarning):
        result = _validate_shap_variance_penalty_factor_parameter(penalty_factor)
    assert result == expected


def test_calculate_number_of_features_to_remove():
    """Test calculation of number of features to remove."""
    # Case 1: More features than minimum required
    result = _calculate_number_of_features_to_remove(
        current_num_of_features=10,
        num_features_to_remove=3,
        min_num_features_to_keep=5,
    )
    assert result == 3  # Can remove all 3 requested features

    # Case 2: Would remove too many features
    result = _calculate_number_of_features_to_remove(
        current_num_of_features=10,
        num_features_to_remove=8,
        min_num_features_to_keep=5,
    )
    assert result == 5  # Can only remove 5 features to maintain minimum

    # Case 3: At the minimum required features
    result = _calculate_number_of_features_to_remove(
        current_num_of_features=5,
        num_features_to_remove=1,
        min_num_features_to_keep=5,
    )
    assert result == 0  # Cannot remove any features


def test_get_current_features_to_remove():
    """Test getting features to remove based on importance and constraints."""
    # Create a mock SHAP importance DataFrame
    shap_importance_df = pd.DataFrame(
        {"importance": [0.5, 0.4, 0.3, 0.2, 0.1]},
        index=["feature1", "feature2", "feature3", "feature4", "feature5"],
    )

    # Test with integer step
    result = _get_current_features_to_remove(
        shap_importance_df=shap_importance_df,
        step=2,
        min_features_to_select=2,
        columns_to_keep=None,
    )
    assert result == ["feature5", "feature4"]  # Remove 2 lowest importance features

    # Test with float step (0.4 * 5 = 2 features)
    result = _get_current_features_to_remove(
        shap_importance_df=shap_importance_df,
        step=0.4,
        min_features_to_select=2,
        columns_to_keep=None,
    )
    assert result == ["feature5", "feature4"]  # Remove 2 lowest importance features

    # Test with columns_to_keep
    result = _get_current_features_to_remove(
        shap_importance_df=shap_importance_df,
        step=3,
        min_features_to_select=2,
        columns_to_keep=["feature4"],
    )
    assert "feature4" not in result  # Should not remove feature4
    assert len(result) == 2  # Should still remove 2 features

    # Test with min_features_to_select constraint
    result = _get_current_features_to_remove(
        shap_importance_df=shap_importance_df,
        step=4,
        min_features_to_select=3,
        columns_to_keep=None,
    )
    assert len(result) == 2  # Should only remove 2 features to maintain minimum of 3

    # Test case where no features should be removed
    result = _get_current_features_to_remove(
        shap_importance_df=shap_importance_df,
        step=1,
        min_features_to_select=5,
        columns_to_keep=None,
    )
    assert result == []  # Cannot remove any features as we're at minimum


@pytest.mark.parametrize(
    "best_method, expected_num_features",
    [
        ("best", 4),  # Highest score
        ("best_coherent", 3),  # Most consistent within threshold
        ("best_parsimonious", 2),  # Fewest features within threshold
    ],
)
def test_get_best_num_features(best_method, expected_num_features):
    """Test getting best number of features using different methods."""
    # Create a mock report DataFrame with controlled metrics
    report_df = pd.DataFrame(
        {
            "num_features": [5, 4, 3, 2, 1],
            "val_metric_mean": [0.75, 0.80, 0.78, 0.77, 0.65],  # Best score at 4 features
            "val_metric_std": [0.08, 0.06, 0.03, 0.07, 0.10],  # Most consistent at 3 features
        },
        index=[1, 2, 3, 4, 5],
    )

    # Get best number of features with standard_error_threshold=0.05
    # This means scores within 0.05 of the best (0.80) will be considered, so 0.75-0.80
    result = _get_best_num_features(
        report_df=report_df,
        best_method=best_method,
        standard_error_threshold=0.05,
        verbose=0,
    )
    assert result == expected_num_features


@pytest.mark.parametrize(
    "best_method, expected_error",
    [
        ("invalid_method", ValueError),  # Invalid method name
        (123, ValueError),  # Invalid type
    ],
)
def test_get_best_num_features_errors(best_method, expected_error):
    """Test errors in get_best_num_features."""
    report_df = pd.DataFrame(
        {
            "num_features": [3, 2, 1],
            "val_metric_mean": [0.8, 0.75, 0.7],
            "val_metric_std": [0.05, 0.04, 0.06],
        },
        index=[1, 2, 3],
    )

    with pytest.raises(expected_error):
        _get_best_num_features(report_df, best_method)


@pytest.mark.parametrize(
    "threshold, expected_error",
    [
        (-1, ValueError),  # Negative threshold
        ("invalid", ValueError),  # Invalid type
    ],
)
def test_get_best_num_features_threshold_errors(threshold, expected_error):
    """Test errors in get_best_num_features with invalid threshold values."""
    report_df = pd.DataFrame(
        {
            "num_features": [3, 2, 1],
            "val_metric_mean": [0.8, 0.75, 0.7],
            "val_metric_std": [0.05, 0.04, 0.06],
        },
        index=[1, 2, 3],
    )

    with pytest.raises(expected_error):
        _get_best_num_features(report_df, "best", standard_error_threshold=threshold)
