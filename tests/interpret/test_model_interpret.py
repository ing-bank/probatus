import numpy as np
import pandas as pd
import pytest

from probatus.interpret import ShapModelInterpreter


@pytest.fixture(scope="function")
def expected_feature_importance():
    return pd.DataFrame(
        {
            "mean_abs_shap_value_test": [0.5, 0.0, 0.0],
            "mean_abs_shap_value_train": [0.5, 0.0, 0.0],
            "mean_shap_value_test": [-0.5, 0.0, 0.0],
            "mean_shap_value_train": [-0.5, 0.0, 0.0],
        },
        index=["col_3", "col_1", "col_2"],
    )


@pytest.fixture(scope="function")
def expected_feature_importance_lin_models():
    return pd.DataFrame(
        {
            "mean_abs_shap_value_test": [0.4, 0.0, 0.0],
            "mean_abs_shap_value_train": [0.4, 0.0, 0.0],
            "mean_shap_value_test": [-0.4, 0.0, 0.0],
            "mean_shap_value_train": [-0.4, 0.0, 0.0],
        },
        index=["col_3", "col_1", "col_2"],
    )


def test_shap_interpret(fitted_tree, X_train, y_train, X_test, y_test, expected_feature_importance, random_state):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.mean(np.abs(shap_interpret.shap_values_test), axis=0) == [0, 0, 0.5]).all()
    assert (np.mean(np.abs(shap_interpret.shap_values_train), axis=0) == [0, 0, 0.5]).all()

    importance_df, train_auc, test_auc = shap_interpret.compute(return_scores=True)

    pd.testing.assert_frame_equal(expected_feature_importance, importance_df)
    assert train_auc == 1
    assert test_auc == pytest.approx(0.833, 0.01)

    # Check if plots work for such dataset
    ax1 = shap_interpret.plot("importance", target_set="test", show=False)
    ax2 = shap_interpret.plot("summary", target_set="test", show=False)
    ax3 = shap_interpret.plot("dependence", target_columns="col_3", target_set="test", show=False)
    ax4 = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    ax5 = shap_interpret.plot("importance", target_set="train", show=False)
    ax6 = shap_interpret.plot("summary", target_set="train", show=False)
    ax7 = shap_interpret.plot("dependence", target_columns="col_3", target_set="train", show=False)
    ax8 = shap_interpret.plot("sample", samples_index=X_train.index.tolist()[0:2], target_set="train", show=False)
    assert not (isinstance(ax1, list))
    assert not (isinstance(ax2, list))
    assert isinstance(ax3, list) and len(ax4) == 2
    assert isinstance(ax4, list) and len(ax4) == 2
    assert not (isinstance(ax5, list))
    assert not (isinstance(ax6, list))
    assert isinstance(ax7, list) and len(ax7) == 2
    assert isinstance(ax8, list) and len(ax8) == 2


def test_shap_interpret_lin_models(
    fitted_logistic_regression, X_train, y_train, X_test, y_test, expected_feature_importance_lin_models, random_state
):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_logistic_regression, random_state=random_state)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_test), axis=0), 2) == [0, 0, 0.4]).all()
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_train), axis=0), 2) == [0, 0, 0.4]).all()

    importance_df, train_auc, test_auc = shap_interpret.compute(return_scores=True)
    importance_df = importance_df.round(2)

    pd.testing.assert_frame_equal(expected_feature_importance_lin_models, importance_df)
    assert train_auc == 1
    assert test_auc == pytest.approx(0.833, 0.01)

    # Check if plots work for such dataset
    ax1 = shap_interpret.plot("importance", target_set="test", show=False)
    ax2 = shap_interpret.plot("summary", target_set="test", show=False)
    ax3 = shap_interpret.plot("dependence", target_columns="col_3", target_set="test", show=False)
    ax4 = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    ax5 = shap_interpret.plot("importance", target_set="train", show=False)
    ax6 = shap_interpret.plot("summary", target_set="train", show=False)
    ax7 = shap_interpret.plot("dependence", target_columns="col_3", target_set="train", show=False)
    ax8 = shap_interpret.plot("sample", samples_index=X_train.index.tolist()[0:2], target_set="train", show=False)
    assert not (isinstance(ax1, list))
    assert not (isinstance(ax2, list))
    assert isinstance(ax3, list) and len(ax4) == 2
    assert isinstance(ax4, list) and len(ax4) == 2
    assert not (isinstance(ax5, list))
    assert not (isinstance(ax6, list))
    assert isinstance(ax7, list) and len(ax7) == 2
    assert isinstance(ax8, list) and len(ax8) == 2


def test_shap_interpret_fit_compute_lin_models(
    fitted_logistic_regression, X_train, y_train, X_test, y_test, expected_feature_importance_lin_models, random_state
):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_logistic_regression, random_state=random_state)
    importance_df = shap_interpret.fit_compute(X_train, X_test, y_train, y_test, class_names=class_names)
    importance_df = importance_df.round(2)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1

    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_test), axis=0), 2) == [0, 0, 0.4]).all()
    assert (np.round(np.mean(np.abs(shap_interpret.shap_values_train), axis=0), 2) == [0, 0, 0.4]).all()

    pd.testing.assert_frame_equal(expected_feature_importance_lin_models, importance_df)


def test_shap_interpret_fit_compute(
    fitted_tree, X_train, y_train, X_test, y_test, expected_feature_importance, random_state
):
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)
    importance_df = shap_interpret.fit_compute(X_train, X_test, y_train, y_test, class_names=class_names)

    assert shap_interpret.class_names == class_names
    assert shap_interpret.train_score == 1
    assert shap_interpret.test_score == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.mean(np.abs(shap_interpret.shap_values_test), axis=0) == [0, 0, 0.5]).all()
    assert (np.mean(np.abs(shap_interpret.shap_values_train), axis=0) == [0, 0, 0.5]).all()

    pd.testing.assert_frame_equal(expected_feature_importance, importance_df)


def test_shap_interpret_complex_data(complex_data_split_with_categorical, complex_fitted_lightgbm, random_state):
    class_names = ["neg", "pos"]
    X_train, X_test, y_train, y_test = complex_data_split_with_categorical

    shap_interpret = ShapModelInterpreter(complex_fitted_lightgbm, verbose=1, random_state=random_state)
    importance_df = shap_interpret.fit_compute(
        X_train, X_test, y_train, y_test, class_names=class_names, approximate=False, check_additivity=False
    )

    assert shap_interpret.class_names == class_names
    assert importance_df.shape[0] == X_train.shape[1]

    # Check if plots work for such dataset
    ax1 = shap_interpret.plot("importance", target_set="test", show=False)
    ax2 = shap_interpret.plot("summary", target_set="test", show=False)
    ax3 = shap_interpret.plot("dependence", target_columns="f2_missing", target_set="test", show=False)
    ax4 = shap_interpret.plot("sample", samples_index=X_test.index.tolist()[0:2], target_set="test", show=False)
    ax5 = shap_interpret.plot("importance", target_set="train", show=False)
    ax6 = shap_interpret.plot("summary", target_set="train", show=False)
    ax7 = shap_interpret.plot("dependence", target_columns="f2_missing", target_set="train", show=False)
    ax8 = shap_interpret.plot("sample", samples_index=X_train.index.tolist()[0:2], target_set="train", show=False)
    assert not (isinstance(ax1, list))
    assert not (isinstance(ax2, list))
    assert isinstance(ax3, list) and len(ax4) == 2
    assert isinstance(ax4, list) and len(ax4) == 2
    assert not (isinstance(ax5, list))
    assert not (isinstance(ax6, list))
    assert isinstance(ax7, list) and len(ax7) == 2
    assert isinstance(ax8, list) and len(ax8) == 2
