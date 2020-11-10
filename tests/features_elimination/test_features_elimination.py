from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np
import pandas as pd
from probatus.features_elimination import ShapRFECV
from unittest.mock import patch

@pytest.fixture(scope='function')
def X():
    return pd.DataFrame({'col_1': [1, 1, 1, 1, 1, 1, 1, 0],
                         'col_2': [0, 0, 0, 0, 0, 0, 0, 1],
                         'col_3': [1, 0, 1, 0, 1, 0, 1, 0]}, index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope='function')
def y():
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])


def test_shap_bfe_grid(X, y):

    clf = DecisionTreeClassifier(max_depth=1)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [1, 2]
    }

    shap_elimination = ShapRFECV(clf, search_space=param_grid, search_schema='grid',
                                 step=0.8, cv=2, scoring='roc_auc', n_jobs=4)

    report = shap_elimination.fit_compute(X, y)

    assert shap_elimination.fitted == True
    shap_elimination._check_if_fitted

    assert report.shape[0] == 2
    assert shap_elimination.get_reduced_features_set(1) == ['col_3']

    ax1 = shap_elimination.plot('performance', show=False)
    ax2 = shap_elimination.plot('parameter', param_names=['criterion', 'min_samples_split'], show=False)

    assert not(isinstance(ax1, list))
    assert isinstance(ax2, list) and len(ax2) == 2


def test_shap_bfe_random(X, y):

    clf = DecisionTreeClassifier(max_depth=1)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [1, 2]
    }

    shap_elimination = ShapRFECV(clf, search_space=param_grid, search_schema='random', n_iter=3, random_state=1,
                                 step=1, cv=2, scoring='roc_auc', n_jobs=4)

    shap_elimination.fit(X, y)

    assert shap_elimination.fitted == True
    shap_elimination._check_if_fitted

    report = shap_elimination.compute()

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ['col_3']

    ax1 = shap_elimination.plot('performance', show=False)
    ax2 = shap_elimination.plot('parameter', param_names=['criterion', 'min_samples_split'], show=False)

    assert not(isinstance(ax1, list))
    assert isinstance(ax2, list) and len(ax2) == 2


def test_calculate_number_of_features_to_remove():
    assert 3 == ShapRFECV._calculate_number_of_features_to_remove(current_num_of_features=10,
                                                                  num_features_to_remove=3,
                                                                  min_num_features_to_keep=5)
    assert 3 == ShapRFECV._calculate_number_of_features_to_remove(current_num_of_features=8,
                                                                  num_features_to_remove=5,
                                                                  min_num_features_to_keep=5)
    assert 0 == ShapRFECV._calculate_number_of_features_to_remove(current_num_of_features=5,
                                                                  num_features_to_remove=1,
                                                                  min_num_features_to_keep=5)
    assert 4 == ShapRFECV._calculate_number_of_features_to_remove(current_num_of_features=5,
                                                                  num_features_to_remove=7,
                                                                  min_num_features_to_keep=1)

def test_preprocess_data(X):
    X['col_static'] = 1
    X['col_categorical'] = X['col_3'].apply(lambda x: str(x))
    X['col_missing'] = X['col_3'].apply(lambda x: x if x < 0.5 else np.nan)

    X_output = ShapRFECV._preprocess_data(X)

    # Check if categoricals correctly handled
    assert X_output['col_categorical'].dtype.name == 'category'

    # Check if static feature removed
    assert 'col_static' not in X_output.columns.tolist()
    assert X_output.shape[1] == X.shape[1] - 1


