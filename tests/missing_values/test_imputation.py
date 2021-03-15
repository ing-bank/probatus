# Code to test the imputation strategies.
from probatus.missing_values.imputation import ImputationSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
import pandas as pd
import numpy as np
import pytest
import os


@pytest.fixture(scope="function")
def X():
    """
    Fixture.
    """
    return pd.DataFrame(
        {
            "col_1": [1, np.nan, 1, 1, np.nan, 1, 1, 0, 1, 1],
            "col_2": [0, 0, 0, np.nan, 0, 0, 0, 1, 0, 0],
            "col_3": [1, 0, np.nan, 0, 1, np.nan, 1, 0, 1, 1],
            "col_4": ["A", "B", "A", np.nan, "B", np.nan, "C", "A", "B", "C"],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )


@pytest.fixture(scope="function")
def y():
    """
    Fixture.
    """
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture(scope="function")
def strategies():
    """
    Test strategies.
    """
    return {
        "Simple Median Imputer": SimpleImputer(strategy="median", add_indicator=True),
        "Simple Mean Imputer": SimpleImputer(strategy="mean", add_indicator=True),
        "Iterative Imputer": IterativeImputer(add_indicator=True, n_nearest_features=5, sample_posterior=True),
        "KNN": KNNImputer(n_neighbors=3),
    }


def test_imputation_linear(X, y, strategies, capsys):
    """
    Test imputation linear.
    """
    # Initialize the classifier
    clf = LogisticRegression()
    cmp = ImputationSelector(clf=clf, strategies=strategies, cv=3, model_na_support=False)
    report = cmp.fit_compute(X, y)
    _ = cmp.plot(show=False)

    assert cmp.fitted
    cmp._check_if_fitted()
    assert report.shape[0] == 4

    # Check if there is any prints
    out, _ = capsys.readouterr()
    assert len(out) == 0


def test_imputation_bagging(X, y, strategies, capsys):
    """
    Test bagging.
    """
    # Initialize the classifier
    clf = RandomForestClassifier()
    cmp = ImputationSelector(clf=clf, strategies=strategies, cv=3, model_na_support=False)
    report = cmp.fit_compute(X, y)
    _ = cmp.plot(show=False)

    assert cmp.fitted
    cmp._check_if_fitted()
    assert report.shape[0] == 4

    # Check if there is any prints
    out, _ = capsys.readouterr()
    assert len(out) == 0


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_imputation_boosting(X, y, strategies, complex_lightgbm, capsys):
    """
    Test boosting.
    """
    # Initialize the classifier
    clf = complex_lightgbm
    cmp = ImputationSelector(clf=clf, strategies=strategies, cv=3, model_na_support=True)
    report = cmp.fit_compute(X, y)
    _ = cmp.plot(show=False)

    assert cmp.fitted
    cmp._check_if_fitted()
    assert report.shape[0] == 5

    # Check if there is any prints
    out, _ = capsys.readouterr()
    assert len(out) == 0