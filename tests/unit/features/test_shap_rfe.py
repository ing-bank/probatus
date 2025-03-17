import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from probatus.features import ShapRFECV
from probatus.core import NotFittedError


# Validation tests
def test_validate_step():
    """Test the _validate_step method."""
    # Valid inputs
    assert ShapRFECV._validate_step(1) == 1
    assert ShapRFECV._validate_step(0.5) == 0.5

    # Invalid inputs
    with pytest.raises(ValueError):
        ShapRFECV._validate_step(0)
    with pytest.raises(ValueError):
        ShapRFECV._validate_step(-1)
    with pytest.raises(ValueError):
        ShapRFECV._validate_step("invalid")


def test_validate_min_features():
    """Test the _validate_min_features method."""
    # Valid inputs
    assert ShapRFECV._validate_min_features(1) == 1
    assert ShapRFECV._validate_min_features(5) == 5

    # Invalid inputs
    with pytest.raises(ValueError):
        ShapRFECV._validate_min_features(0)
    with pytest.raises(ValueError):
        ShapRFECV._validate_min_features(-1)
    with pytest.raises(ValueError):
        ShapRFECV._validate_min_features(1.5)
    with pytest.raises(ValueError):
        ShapRFECV._validate_min_features("invalid")


def test_shap_variance_penalty_factor():
    """Test the shap_variance_penalty_factor parameter validation."""
    # Test with negative value
    with pytest.warns(UserWarning, match="shap_variance_penalty_factor must be None, int or float"):
        assert ShapRFECV._validate_shap_variance_penalty_factor(-1) == 0

    # Test with invalid type
    with pytest.warns(UserWarning, match="shap_variance_penalty_factor must be None, int or float"):
        assert ShapRFECV._validate_shap_variance_penalty_factor("invalid") == 0

    # Test with valid values
    assert ShapRFECV._validate_shap_variance_penalty_factor(0) == 0
    assert ShapRFECV._validate_shap_variance_penalty_factor(0.5) == 0.5
    assert ShapRFECV._validate_shap_variance_penalty_factor(1.0) == 1.0


# Feature elimination calculation tests
def test_calculate_number_of_features_to_remove():
    """Test the _calculate_number_of_features_to_remove method."""
    # Normal case - remove 3 features
    assert 3 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=10, num_features_to_remove=3, min_num_features_to_keep=5
    )

    # Limit case - can only remove 3 features to maintain minimum
    assert 3 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=8, num_features_to_remove=5, min_num_features_to_keep=5
    )

    # Boundary case - can't remove any features
    assert 0 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=5, num_features_to_remove=1, min_num_features_to_keep=5
    )

    # Remove all but minimum required
    assert 4 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=5, num_features_to_remove=7, min_num_features_to_keep=1
    )


def test_get_current_features_to_remove(mock_shap_elimination, shap_importance_df):
    """Test the _get_current_features_to_remove method."""
    # Test with integer step
    mock_shap_elimination.step = 1
    mock_shap_elimination.min_features_to_select = 1
    features_to_remove = mock_shap_elimination._get_current_features_to_remove(shap_importance_df)
    assert features_to_remove == ["col_3"]  # Lowest importance feature

    # Test with float step (33% of features)
    mock_shap_elimination.step = 0.33
    features_to_remove = mock_shap_elimination._get_current_features_to_remove(shap_importance_df)
    assert features_to_remove == ["col_3"]  # 33% of 3 features = 1 feature

    # Test with columns_to_keep
    features_to_remove = mock_shap_elimination._get_current_features_to_remove(
        shap_importance_df, columns_to_keep=["col_3"]
    )
    assert features_to_remove == ["col_2"]  # col_3 is protected, so remove col_2

    # Test with min_features_to_select limiting removal
    mock_shap_elimination.step = 3
    mock_shap_elimination.min_features_to_select = 2
    features_to_remove = mock_shap_elimination._get_current_features_to_remove(shap_importance_df)
    assert features_to_remove == ["col_3"]  # Can only remove 1 feature to keep 2

    # Test with no features to remove
    mock_shap_elimination.min_features_to_select = 3
    features_to_remove = mock_shap_elimination._get_current_features_to_remove(shap_importance_df)
    assert features_to_remove == []  # Can't remove any features


def test_filter_and_identify_features_based_on_importance(mock_shap_elimination, shap_importance_df):
    """Test the _filter_and_identify_features_based_on_importance method."""
    # Set up the mock
    mock_shap_elimination.step = 1
    mock_shap_elimination.min_features_to_select = 1

    # Test normal case
    current_features_set = ["col_1", "col_2", "col_3"]
    remaining, removed = mock_shap_elimination._filter_and_identify_features_based_on_importance(
        shap_importance_df, None, current_features_set
    )
    assert remaining == ["col_1", "col_2"]
    assert removed == ["col_3"]

    # Test with columns_to_keep
    remaining, removed = mock_shap_elimination._filter_and_identify_features_based_on_importance(
        shap_importance_df, ["col_3"], current_features_set
    )
    assert remaining == ["col_1", "col_3"]  # col_3 is kept despite low importance
    assert removed == ["col_2"]


# Feature support and ranking tests
def test_get_feature_support(mock_shap_elimination):
    """Test the _get_feature_support method."""
    feature_names_selected = ["col_1", "col_3"]
    support = mock_shap_elimination._get_feature_support(feature_names_selected)
    assert support == [True, False, True]  # col_1 and col_3 are True, col_2 is False


def test_get_feature_ranking(mock_shap_elimination, feature_ranking_report_df):
    """Test the _get_feature_ranking method."""
    mock_shap_elimination.report_df = feature_ranking_report_df

    ranking = mock_shap_elimination._get_feature_ranking()
    # col_1 was never eliminated (rank 0)
    # col_2 was eliminated in round 2 (rank 1)
    # col_3 was eliminated in round 1 (rank 2)
    assert ranking == [0, 1, 2]


def test_get_feature_names(mock_shap_elimination, feature_selection_report_df):
    """Test the _get_feature_names method."""
    mock_shap_elimination.report_df = feature_selection_report_df

    assert mock_shap_elimination._get_feature_names(3) == ["col_1", "col_2", "col_3"]
    assert mock_shap_elimination._get_feature_names(2) == ["col_1", "col_2"]
    assert mock_shap_elimination._get_feature_names(1) == ["col_1"]

    with pytest.raises(ValueError):
        mock_shap_elimination._get_feature_names(4)  # Not in report_df


# Model compatibility tests
def test_check_if_model_is_compatible_with_early_stopping(logistic_regression, xgb_model, lgbm_regressor):
    """Test the _check_if_model_is_compatible_with_early_stopping method."""
    # Test if model is not compatible with early stopping
    assert not ShapRFECV._check_if_model_is_compatible_with_early_stopping(logistic_regression)

    # Test if model is compatible with early stopping
    assert ShapRFECV._check_if_model_is_compatible_with_early_stopping(xgb_model)
    assert ShapRFECV._check_if_model_is_compatible_with_early_stopping(lgbm_regressor)


# Feature selection strategy tests
def test_get_best_num_features(mock_shap_elimination, feature_selection_report_df):
    """Test the _get_best_num_features method."""
    mock_shap_elimination.report_df = feature_selection_report_df

    # Test 'best' method - should return features from highest val_metric_mean
    best_num = mock_shap_elimination._get_best_num_features("best")
    assert best_num == 2  # Round 2 has highest val_metric_mean

    # Test 'best_coherent' method - should return features with lowest std within threshold
    best_coherent_num = mock_shap_elimination._get_best_num_features("best_coherent", standard_error_threshold=0.1)
    assert best_coherent_num == 3  # Round 1 has lowest std within 0.1 of best score

    # Test 'best_parsimonious' method - should return fewest features within threshold
    best_parsimonious_num = mock_shap_elimination._get_best_num_features(
        "best_parsimonious", standard_error_threshold=0.1
    )
    assert best_parsimonious_num == 1  # Round 3 has fewest features within 0.1 of best score

    # Test invalid method
    with pytest.raises(ValueError):
        mock_shap_elimination._get_best_num_features("invalid_method")

    # Test invalid threshold
    with pytest.raises(ValueError):
        mock_shap_elimination._get_best_num_features("best", standard_error_threshold=-1)


def test_get_reduced_features_set_with_auto_selection(mock_shap_elimination, auto_selection_report_df):
    """Test get_reduced_features_set with automatic feature selection strategies."""
    mock_shap_elimination.report_df = auto_selection_report_df
    mock_shap_elimination.column_names = ["col_1", "col_2", "col_3", "col_4", "col_5"]

    # Test "best" strategy
    best_features = mock_shap_elimination.get_reduced_features_set(num_features="best")
    assert best_features == ["col_1", "col_2", "col_3"]  # Round 3 has highest val_metric_mean

    # Test "best_coherent" strategy
    best_coherent_features = mock_shap_elimination.get_reduced_features_set(
        num_features="best_coherent", standard_error_threshold=0.1
    )
    assert best_coherent_features == ["col_1", "col_2", "col_3"]  # Round 3 has lowest std within threshold

    # Test "best_parsimonious" strategy
    best_parsimonious_features = mock_shap_elimination.get_reduced_features_set(
        num_features="best_parsimonious", standard_error_threshold=0.2
    )
    assert best_parsimonious_features == ["col_1"]  # Round 5 has fewest features within threshold


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


def test_early_stopping_shap_rfecv_init(random_state_42):
    """Test initialization of EarlyStoppingShapRFECV."""
    model = LogisticRegression(random_state=random_state_42)

    # Mock the compatibility check to return True
    original_check = ShapRFECV._check_if_model_is_compatible_with_early_stopping
    ShapRFECV._check_if_model_is_compatible_with_early_stopping = staticmethod(lambda model: True)

    try:
        # Test default parameters
        early_stopping = ShapRFECV(model, early_stopping_rounds=5, eval_metric="auc")
        assert early_stopping.early_stopping_rounds == 5
        assert early_stopping.eval_metric == "auc"

        # Test custom parameters
        early_stopping = ShapRFECV(
            model, step=0.5, min_features_to_select=2, early_stopping_rounds=10, eval_metric="logloss"
        )
        assert early_stopping.step == 0.5
        assert early_stopping.min_features_to_select == 2
        assert early_stopping.early_stopping_rounds == 10
        assert early_stopping.eval_metric == "logloss"
    finally:
        # Restore the original method
        ShapRFECV._check_if_model_is_compatible_with_early_stopping = original_check


# API tests
def test_check_if_fitted(mock_shap_elimination):
    """Test the _check_if_fitted method."""
    # Should not raise error when fitted=True
    mock_shap_elimination._check_if_fitted()

    # Should raise error when fitted=False
    mock_shap_elimination.fitted = False
    with pytest.raises(NotFittedError):
        mock_shap_elimination._check_if_fitted()


def test_report_current_results(mock_shap_elimination, results_report_df):
    """Test the _report_current_results method."""
    mock_shap_elimination.report_df = results_report_df

    mock_shap_elimination._report_current_results(
        round_number=1,
        current_features_set=["col_1", "col_2", "col_3"],
        features_to_remove=["col_3"],
        train_metric_mean=0.8,
        train_metric_std=0.1,
        val_metric_mean=0.75,
        val_metric_std=0.15,
    )

    # Check that report_df was updated correctly
    assert len(mock_shap_elimination.report_df) == 1
    assert 1 in mock_shap_elimination.report_df.index
    assert mock_shap_elimination.report_df.loc[1, "num_features"] == 3
    assert mock_shap_elimination.report_df.loc[1, "features_set"] == ["col_1", "col_2", "col_3"]
    assert mock_shap_elimination.report_df.loc[1, "eliminated_features"] == ["col_3"]
    assert mock_shap_elimination.report_df.loc[1, "train_metric_mean"] == 0.8
    assert mock_shap_elimination.report_df.loc[1, "train_metric_std"] == 0.1
    assert mock_shap_elimination.report_df.loc[1, "val_metric_mean"] == 0.75
    assert mock_shap_elimination.report_df.loc[1, "val_metric_std"] == 0.15


def test_get_reduced_features_set_return_types(mock_shap_elimination, feature_selection_report_df):
    """Test different return types of get_reduced_features_set method."""
    mock_shap_elimination.report_df = feature_selection_report_df
    mock_shap_elimination.report_df["eliminated_features"] = [[], ["col_3"], ["col_2"]]

    # Test feature_names return type (default)
    feature_names = mock_shap_elimination.get_reduced_features_set(num_features=2)
    assert feature_names == ["col_1", "col_2"]

    # Test support return type
    support = mock_shap_elimination.get_reduced_features_set(num_features=2, return_type="support")
    assert support == [True, True, False]

    # Test ranking return type
    ranking = mock_shap_elimination.get_reduced_features_set(num_features=2, return_type="ranking")
    assert isinstance(ranking, list)

    # Test invalid return type
    with pytest.raises(ValueError):
        mock_shap_elimination.get_reduced_features_set(num_features=2, return_type="invalid")


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


def test_get_fit_params(random_state_42):
    """Test the _get_fit_params method."""
    model = LogisticRegression(random_state=random_state_42)
    shap_elimination = ShapRFECV(model, random_state=random_state_42)
    shap_elimination.early_stopping_rounds = 5
    shap_elimination.eval_metric = "auc"

    X_train = pd.DataFrame({"col_1": [1, 0], "col_2": [0, 1], "col_3": [1, 0]})
    y_train = pd.Series([1, 0])
    X_val = pd.DataFrame({"col_1": [1, 0], "col_2": [0, 1], "col_3": [1, 0]})
    y_val = pd.Series([1, 0])
    sample_weight = pd.Series([1, 1])
    train_index = np.array([0, 1])
    val_index = np.array([0, 1])

    # Test with unsupported model
    with pytest.raises(ValueError):
        shap_elimination._get_fit_params(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            train_index=train_index,
            val_index=val_index,
        )
