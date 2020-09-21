from probatus.sample_similarity import BaseResemblanceModel, SHAPImportanceResemblance, PermutationImportanceResemblance
from probatus.utils import NotFittedError

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture(scope='function')
def X1():
    return pd.DataFrame({'col_1': [1, 1, 1, 1],
                         'col_2': [0, 0, 0, 0],
                         'col_3': [0, 0, 0, 0]}, index=[1, 2, 3, 4])


@pytest.fixture(scope='function')
def X2():
    return pd.DataFrame({'col_1': [0, 0, 0, 0],
                         'col_2': [0, 0, 0, 0],
                         'col_3': [0, 0, 0, 0]}, index=[1, 2, 3, 4])


def test_base_class(X1, X2):
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = BaseResemblanceModel(clf, test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_auc, test_auc = rm.fit_compute(X1, X2, return_auc=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_auc == 1
    assert test_auc == 1
    assert actual_report is None

    # Check data splits if correct
    actual_X_train, actual_X_test, actual_y_train, actual_y_test = rm.get_data_splits()

    assert actual_X_train.shape == (4, 3)
    assert actual_X_test.shape == (4, 3)
    assert len(actual_y_train) == 4
    assert len(actual_y_test) == 4

    # Check if stratified
    assert np.sum(actual_y_train) == 2
    assert np.sum(actual_y_test) == 2

    # Check if index is correct
    assert len(rm.X.index.unique()) == 8
    assert list(rm.X.index) == list(rm.y.index)


def test_shap_resemblance_class(X1, X2):
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = SHAPImportanceResemblance(clf, test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_auc, test_auc = rm.fit_compute(X1, X2, return_auc=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_auc == 1
    assert test_auc == 1

    # Check report shape
    assert actual_report.shape == (3, 2)
    # Check if it is sorted by importance
    print(actual_report)
    assert actual_report.iloc[0].name == 'col_1'
    # Check report values
    assert actual_report.loc['col_1']['mean_abs_shap_value'] > 0
    assert actual_report.loc['col_1']['mean_shap_value'] == 0
    assert actual_report.loc['col_2']['mean_abs_shap_value'] == 0
    assert actual_report.loc['col_2']['mean_shap_value'] == 0
    assert actual_report.loc['col_3']['mean_abs_shap_value'] == 0
    assert actual_report.loc['col_3']['mean_shap_value'] == 0

    actual_shap_values_test = rm.get_shap_values()
    assert actual_shap_values_test.shape == (4, 3)

    # Run plots
    rm.plot(plot_type='bar')
    rm.plot(plot_type='dot')


def test_permutation_resemblance_class(X1, X2):
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = PermutationImportanceResemblance(clf, test_prc=0.5, n_jobs=1, random_state=42, iterations=20)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_auc, test_auc = rm.fit_compute(X1, X2, return_auc=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_auc == 1
    assert test_auc == 1

    # Check report shape
    assert actual_report.shape == (3, 2)
    # Check if it is sorted by importance
    print(actual_report)
    assert actual_report.iloc[0].name == 'col_1'
    # Check report values
    assert actual_report.loc['col_1']['mean_importance'] > 0
    assert actual_report.loc['col_1']['std_importance'] > 0
    assert actual_report.loc['col_2']['mean_importance'] == 0
    assert actual_report.loc['col_2']['std_importance'] == 0
    assert actual_report.loc['col_3']['mean_importance'] == 0
    assert actual_report.loc['col_3']['std_importance'] == 0

    rm.plot()
    # Check plot size
    fig = plt.gcf()
    size = fig.get_size_inches()
    assert size[0] == 10 and size[1] == 2.5


def test_base_class_same_data(X1):
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = BaseResemblanceModel(clf, test_prc=0.5, n_jobs=1, random_state=42)

    actual_report, train_auc, test_auc = rm.fit_compute(X1, X1, return_auc=True)

    assert train_auc == 0.5
    assert test_auc == 0.5
    assert actual_report is None