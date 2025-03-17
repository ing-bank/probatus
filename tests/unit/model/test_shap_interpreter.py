import numpy as np
import pandas as pd

from probatus.model import ShapModelInterpreter
from probatus.utils import Scorer


def test_init_parameters(random_state_42):
    """Test initialization parameters of ShapModelInterpreter."""
    # Test with default parameters
    shap_interpret = ShapModelInterpreter(model="dummy_model")
    assert shap_interpret.model == "dummy_model"
    assert shap_interpret.scorer.metric_name == "roc_auc"
    assert shap_interpret.verbose == 0
    assert shap_interpret.random_state is None
    assert shap_interpret.fitted is False

    # Test with custom parameters
    custom_scorer = Scorer(metric_name="custom_metric", custom_scorer=lambda y_true, y_pred: 0.5)
    shap_interpret = ShapModelInterpreter(
        model="dummy_model", scoring=custom_scorer, verbose=1, random_state=random_state_42
    )
    assert shap_interpret.model == "dummy_model"
    assert shap_interpret.scorer.metric_name == "custom_metric"
    assert shap_interpret.verbose == 1
    assert shap_interpret.random_state == random_state_42
    assert shap_interpret.fitted is False


def test_fit_with_custom_column_names(fitted_tree, X_train, y_train, X_test, y_test, random_state):
    """Test the fit method with custom column names."""
    # Need to rename DataFrame columns to match custom_column_names
    X_train_renamed = X_train.copy()
    X_train_renamed.columns = ["feature_1", "feature_2", "feature_3"]
    X_test_renamed = X_test.copy()
    X_test_renamed.columns = ["feature_1", "feature_2", "feature_3"]

    custom_column_names = ["feature_1", "feature_2", "feature_3"]

    # Need to retrain the model with the renamed features to avoid scikit-learn feature name checks
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=1, random_state=random_state).fit(X_train_renamed, y_train)

    shap_interpret = ShapModelInterpreter(model, random_state=random_state)
    shap_interpret.fit(X_train_renamed, X_test_renamed, y_train, y_test, column_names=custom_column_names)

    assert shap_interpret.column_names == custom_column_names


def test_compute_logistic_regression(
    fitted_logistic_regression,
    X_train,
    y_train,
    X_test,
    y_test,
    expected_feature_importance_lin_models,
    random_state,
):
    """Test the compute method with a logistic regression model."""
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_logistic_regression, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    # Test compute with return_scores
    importance_df, train_auc, test_auc = shap_interpret.compute(return_scores=True)

    # Use approximate equality with appropriate tolerance
    pd.testing.assert_frame_equal(
        expected_feature_importance_lin_models.round(2), importance_df.round(2), check_exact=False, rtol=1e-2
    )

    assert train_auc == 1
    assert test_auc == 1.0


def test_fit_compute_logistic_regression(
    fitted_logistic_regression,
    X_train,
    y_train,
    X_test,
    y_test,
    expected_feature_importance_lin_models,
    random_state,
):
    """Test the fit_compute method with a logistic regression model."""
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_logistic_regression, random_state=random_state)
    importance_df = shap_interpret.fit_compute(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.fitted is True
    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == 1.0

    # Check expected shap values
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_test), axis=0), 2) == [0, 0, 0.53]).all()
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_train), axis=0), 2) == [0, 0, 0.4]).all()

    pd.testing.assert_frame_equal(
        expected_feature_importance_lin_models.round(2), importance_df.round(2), check_exact=False, rtol=1e-2
    )


def test_fit_compute_with_shap_kwargs(fitted_tree, X_train, y_train, X_test, y_test, random_state):
    """Test the fit_compute method with SHAP kwargs."""
    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    importance_df = shap_interpret.fit_compute(
        X_train, X_test, y_train, y_test, approximate=True, check_additivity=False
    )

    assert shap_interpret.fitted is True
    assert importance_df is not None


def test_prepare_target_columns(fitted_tree, X_train, y_train, X_test, y_test, random_state):
    """Test the _prepare_target_columns method."""
    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test)

    # Test with string
    columns = shap_interpret._prepare_target_columns("col_1")
    assert columns == ["col_1"]

    # Test with list
    columns = shap_interpret._prepare_target_columns(["col_1", "col_3"])
    assert columns == ["col_1", "col_3"]

    # Test with None (should return all columns)
    columns = shap_interpret._prepare_target_columns(None)
    assert set(columns) == {"col_1", "col_2", "col_3"}


def test_select_target_dataset(fitted_tree, X_train, y_train, X_test, y_test, random_state):
    """Test the _select_target_dataset method."""
    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test)

    # Test train set
    train_data = shap_interpret._select_target_dataset("train")
    assert train_data["X"] is shap_interpret.X_train
    assert train_data["shap_values"] is shap_interpret.shap_values_train
    assert train_data["expected_value"] == shap_interpret.expected_value_train
    assert train_data["tdp"] is shap_interpret.tdp_train

    # Test test set
    test_data = shap_interpret._select_target_dataset("test")
    assert test_data["X"] is shap_interpret.X_test
    assert test_data["shap_values"] is shap_interpret.shap_values_test
    assert test_data["expected_value"] == shap_interpret.expected_value_test
    assert test_data["tdp"] is shap_interpret.tdp_test
