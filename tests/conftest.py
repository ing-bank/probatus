from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


@pytest.fixture(scope="function")
def random_state():
    """
    Fixture to automatically provide a random state.
    """
    RANDOM_STATE = 0

    return RANDOM_STATE


@pytest.fixture(scope="function")
def random_state_42():
    """
    Fixture to automatically provide a random state.
    """
    RANDOM_STATE = 42

    return RANDOM_STATE


@pytest.fixture(scope="function")
def random_state_1234():
    """
    Fixture to automatically provide a random state.
    """
    RANDOM_STATE = 1234

    return RANDOM_STATE


@pytest.fixture(scope="function")
def random_state_1():
    """
    Fixture to automatically provide a random state.
    """
    RANDOM_STATE = 1

    return RANDOM_STATE


@pytest.fixture(scope="function")
def mock_model():
    """
    Fixture.
    """
    return Mock()


@pytest.fixture(scope="function")
def complex_data(random_state):
    """
    Fixture.
    """

    feature_names = ["f1_categorical", "f2_missing", "f3_static", "f4", "f5"]

    # Prepare two samples
    X, y = make_classification(
        n_samples=50,
        class_sep=0.05,
        n_informative=2,
        n_features=5,
        random_state=random_state,
        n_redundant=2,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X, columns=feature_names)
    X.loc[0:10, "f2_missing"] = np.nan
    return X, y


@pytest.fixture(scope="function")
def complex_data_with_categorical(complex_data):
    X, y = complex_data
    X["f1_categorical"] = X["f1_categorical"].astype(str).astype("category")

    return X, y


@pytest.fixture(scope="function")
def complex_data_split(complex_data, random_state_42):
    """
    Fixture.
    """
    X, y = complex_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_42)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="function")
def complex_data_split_with_categorical(complex_data_split):
    X_train, X_test, y_train, y_test = complex_data_split
    X_train["f1_categorical"] = X_train["f1_categorical"].astype(str).astype("category")
    X_test["f1_categorical"] = X_test["f1_categorical"].astype(str).astype("category")

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="function")
def complex_lightgbm(random_state_42):
    """This fixture allows to reuse the import of the LGBMClassifier class across different tests."""
    model = LGBMClassifier(max_depth=5, num_leaves=11, class_weight="balanced", random_state=random_state_42)
    return model


@pytest.fixture(scope="function")
def complex_fitted_lightgbm(complex_data_split_with_categorical, complex_lightgbm):
    """
    Fixture.
    """
    X_train, _, y_train, _ = complex_data_split_with_categorical

    return complex_lightgbm.fit(X_train, y_train)


@pytest.fixture(scope="function")
def catboost_classifier(random_state):
    """This fixture allows to reuse the import of the CatboostClassifier class across different tests."""
    model = CatBoostClassifier(random_seed=random_state)
    return model


@pytest.fixture(scope="function")
def decision_tree_classifier(random_state):
    """This fixture allows to reuse the import of the DecisionTreeClassifier class across different tests."""
    model = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    return model


@pytest.fixture(scope="function")
def randomized_search_decision_tree_classifier(decision_tree_classifier, random_state):
    """This fixture allows to reuse the import of the DecisionTreeClassifier in combination with a new CV class across different tests."""
    param_grid = {"criterion": ["gini"], "min_samples_split": [1, 2]}
    cv = RandomizedSearchCV(decision_tree_classifier, param_grid, cv=2, n_iter=2, random_state=random_state)
    return cv


@pytest.fixture(scope="function")
def logistic_regression(random_state):
    """This fixture allows to reuse the import of the DecisionTreeClassifier class across different tests."""
    model = LogisticRegression(random_state=random_state)
    return model


@pytest.fixture(scope="function")
def X_train():
    """
    Fixture.
    """
    return pd.DataFrame({"col_1": [1, 1, 1, 1], "col_2": [0, 0, 0, 0], "col_3": [1, 0, 1, 0]}, index=[1, 2, 3, 4])


@pytest.fixture(scope="function")
def y_train():
    """
    Fixture.
    """
    return pd.Series([1, 0, 1, 0], index=[1, 2, 3, 4])


@pytest.fixture(scope="function")
def X_test():
    """
    Fixture.
    """
    return pd.DataFrame({"col_1": [1, 1, 1, 1], "col_2": [0, 0, 0, 0], "col_3": [1, 0, 1, 0]}, index=[5, 6, 7, 8])


@pytest.fixture(scope="function")
def y_test():
    """
    Fixture.
    """
    return pd.Series([0, 0, 1, 0], index=[5, 6, 7, 8])


@pytest.fixture(scope="function")
def fitted_logistic_regression(X_train, y_train, logistic_regression):
    """
    Fixture.
    """
    return logistic_regression.fit(X_train, y_train)


@pytest.fixture(scope="function")
def fitted_tree(X_train, y_train, decision_tree_classifier):
    """
    Fixture.
    """
    return decision_tree_classifier.fit(X_train, y_train)
