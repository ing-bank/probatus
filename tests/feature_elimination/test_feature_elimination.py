from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np
import pandas as pd
from probatus.feature_elimination import ShapRFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import get_scorer

@pytest.fixture(scope='function')
def X():
    return pd.DataFrame({'col_1': [1, 1, 1, 1, 1, 1, 1, 0],
                         'col_2': [0, 0, 0, 0, 0, 0, 0, 1],
                         'col_3': [1, 0, 1, 0, 1, 0, 1, 0]}, index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope='function')
def y():
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])


def test_shap_rfe_randomized_search(X, y):

    clf = DecisionTreeClassifier(max_depth=1)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [1, 2]
    }
    search = RandomizedSearchCV(clf, param_grid, cv=2)

    shap_elimination = ShapRFECV(search, step=0.8, cv=2, scoring='roc_auc', n_jobs=4)

    report = shap_elimination.fit_compute(X, y)

    assert shap_elimination.fitted == True
    shap_elimination._check_if_fitted

    assert report.shape[0] == 2
    assert shap_elimination.get_reduced_features_set(1) == ['col_3']

    ax1 = shap_elimination.plot(show=False)


def test_shap_rfe(X, y):

    clf = DecisionTreeClassifier(max_depth=1)

    shap_elimination = ShapRFECV(clf, random_state=1, step=1, cv=2, scoring='roc_auc', n_jobs=4)

    shap_elimination.fit(X, y)

    assert shap_elimination.fitted == True
    shap_elimination._check_if_fitted

    report = shap_elimination.compute()

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ['col_3']

    ax1 = shap_elimination.plot(show=False)


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


def test_get_feature_shap_values_per_fold(X, y):
    clf = DecisionTreeClassifier(max_depth=1)
    shap_values, train_score, test_score = ShapRFECV._get_feature_shap_values_per_fold(X, y, clf,
                                                                                       train_index=[2, 3, 4, 5, 6, 7],
                                                                                       val_index=[0, 1],
                                                                                       scorer=get_scorer('roc_auc'))
    assert test_score == 1
    assert train_score > 0.9
    assert shap_values.shape == (2, 3)



