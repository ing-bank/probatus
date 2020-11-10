from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np
import pandas as pd
from probatus.interpret import ShapModelInterpreter
from unittest.mock import patch

@pytest.fixture(scope='function')
def X_train():
    return pd.DataFrame({'col_1': [1, 1, 1, 1],
                         'col_2': [0, 0, 0, 0],
                         'col_3': [1, 0, 1, 0]}, index=[1, 2, 3, 4])


@pytest.fixture(scope='function')
def y_train():
    return pd.Series([1, 0, 1, 0], index=[1, 2, 3, 4])


@pytest.fixture(scope='function')
def X_test():
    return pd.DataFrame({'col_1': [1, 1, 1, 1],
                         'col_2': [0, 0, 0, 0],
                         'col_3': [1, 0, 1, 0]}, index=[5, 6, 7, 8])


@pytest.fixture(scope='function')
def y_test():
    return pd.Series([0, 0, 1, 0], index=[5, 6, 7, 8])


@pytest.fixture(scope='function')
def fitted_tree(X_train, y_train):
    return DecisionTreeClassifier(max_depth=1, random_state=1).fit(X_train, y_train)


@pytest.fixture(scope='function')
def expected_feature_importance():
    return pd.DataFrame({
        'mean_abs_shap_value': [0.5, 0., 0.],
        'mean_shap_value': [0., 0., 0.]}, index=['col_3', 'col_1', 'col_2'])


def test_shap_interpret(fitted_tree, X_train, y_train, X_test, y_test, expected_feature_importance):
    class_names = ['neg', 'pos']

    shap_interpret = ShapModelInterpreter(fitted_tree)
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    # Check parameters
    assert shap_interpret.fitted == True
    shap_interpret._check_if_fitted

    assert shap_interpret.class_names == class_names
    assert shap_interpret.auc_train == 1
    assert shap_interpret.auc_test == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.mean(np.abs(shap_interpret.shap_values), axis=0) == [0, 0, 0.5]).all()

    importance_df = shap_interpret.compute()
    pd.testing.assert_frame_equal(expected_feature_importance, importance_df)

    with patch('matplotlib.pyplot.figure') as mock_plt:
        with patch('shap.plots._waterfall.waterfall_legacy'):
            ax1 = shap_interpret.plot('importance')
            ax2 = shap_interpret.plot('summary')
            ax3 =shap_interpret.plot('dependence', target_columns='col_3')
            ax4 = shap_interpret.plot('sample', samples_index=[5, 6])
    assert not(isinstance(ax1, list))
    assert not(isinstance(ax2, list))
    assert not(isinstance(ax3, list))
    assert isinstance(ax4, list) and len(ax4) == 2


def test_shap_interpret_fit_compute(fitted_tree, X_train, y_train, X_test, y_test, expected_feature_importance):
    class_names = ['neg', 'pos']

    shap_interpret = ShapModelInterpreter(fitted_tree)
    importance_df = shap_interpret.fit_compute(X_train, X_test, y_train, y_test, class_names=class_names)

    # Check parameters
    assert shap_interpret.fitted == True
    shap_interpret._check_if_fitted

    assert shap_interpret.class_names == class_names
    assert shap_interpret.auc_train == 1
    assert shap_interpret.auc_test == pytest.approx(0.833, 0.01)

    # Check expected shap values
    assert (np.mean(np.abs(shap_interpret.shap_values), axis=0) == [0, 0, 0.5]).all()

    pd.testing.assert_frame_equal(expected_feature_importance, importance_df)