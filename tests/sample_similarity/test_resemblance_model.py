from probatus.sample_similarity import BaseResemblanceModel, SHAPImportanceResemblance, PermutationImportanceResemblance
from probatus.utils import NotFittedError

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import os
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")


@pytest.fixture(scope="function")
def X1():
    """
    Fixture.
    """
    return pd.DataFrame({"col_1": [1, 1, 1, 1], "col_2": [0, 0, 0, 0], "col_3": [0, 0, 0, 0]}, index=[1, 2, 3, 4])


@pytest.fixture(scope="function")
def X2():
    """
    Fixture.
    """
    return pd.DataFrame({"col_1": [0, 0, 0, 0], "col_2": [0, 0, 0, 0], "col_3": [0, 0, 0, 0]}, index=[1, 2, 3, 4])


def test_base_class(X1, X2):
    """
    Test.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = BaseResemblanceModel(clf, test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_score, test_score = rm.fit_compute(X1, X2, return_scores=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_score == 1
    assert test_score == 1
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

    with pytest.raises(NotImplementedError) as _:
        rm.plot()


def test_base_class_lin_models(X1, X2):
    """
    Test.
    """
    # Test class BaseResemblanceModel for linear models.
    clf = LogisticRegression()
    rm = BaseResemblanceModel(clf, test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_score, test_score = rm.fit_compute(X1, X2, return_scores=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_score == 1
    assert test_score == 1
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

    with pytest.raises(NotImplementedError) as _:
        rm.plot()


def test_shap_resemblance_class(X1, X2):
    """
    Test.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = SHAPImportanceResemblance(clf, test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_score, test_score = rm.fit_compute(X1, X2, return_scores=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_score == 1
    assert test_score == 1

    # Check report shape
    assert actual_report.shape == (3, 2)
    # Check if it is sorted by importance
    assert actual_report.iloc[0].name == "col_1"
    # Check report values
    assert actual_report.loc["col_1"]["mean_abs_shap_value"] > 0
    assert actual_report.loc["col_1"]["mean_shap_value"] >= 0
    assert actual_report.loc["col_2"]["mean_abs_shap_value"] == 0
    assert actual_report.loc["col_2"]["mean_shap_value"] == 0
    assert actual_report.loc["col_3"]["mean_abs_shap_value"] == 0
    assert actual_report.loc["col_3"]["mean_shap_value"] == 0

    actual_shap_values_test = rm.get_shap_values()
    assert actual_shap_values_test.shape == (4, 3)

    # Run plots
    rm.plot(plot_type="bar")
    rm.plot(plot_type="dot")


def test_shap_resemblance_class_lin_models(X1, X2):
    """
    Test.
    """
    # Test SHAP Resemblance Model for linear models.
    clf = LogisticRegression()
    rm = SHAPImportanceResemblance(clf, test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_score, test_score = rm.fit_compute(
        X1, X2, return_scores=True, approximate=True, check_additivity=False
    )

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_score == 1
    assert test_score == 1

    # Check report shape
    assert actual_report.shape == (3, 2)
    # Check if it is sorted by importance
    assert actual_report.iloc[0].name == "col_1"
    # Check report values
    assert actual_report.loc["col_1"]["mean_abs_shap_value"] > 0
    assert actual_report.loc["col_1"]["mean_shap_value"] > 0
    assert actual_report.loc["col_2"]["mean_abs_shap_value"] == 0
    assert actual_report.loc["col_2"]["mean_shap_value"] == 0
    assert actual_report.loc["col_3"]["mean_abs_shap_value"] == 0
    assert actual_report.loc["col_3"]["mean_shap_value"] == 0

    actual_shap_values_test = rm.get_shap_values()
    assert actual_shap_values_test.shape == (4, 3)

    # Run plots
    rm.plot(plot_type="bar")
    rm.plot(plot_type="dot")


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_shap_resemblance_class2(complex_data, complex_lightgbm):
    """
    Test.
    """
    X1, _ = complex_data
    X2 = X1.copy()
    X2["f4"] = X2["f4"] + 100

    rm = SHAPImportanceResemblance(complex_lightgbm, scoring="accuracy", test_prc=0.5, n_jobs=1, random_state=42)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_score, test_score = rm.fit_compute(X1, X2, return_scores=True, class_names=["a", "b"])

    # Check if the X and y within the rm have correct types
    assert rm.X["f1_categorical"].dtype.name == "category"
    for num_column in ["f2_missing", "f3_static", "f4", "f5"]:
        assert is_numeric_dtype(rm.X[num_column])

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_score == pytest.approx(1, 0.05)
    assert test_score == pytest.approx(1, 0.05)

    # Check report shape
    assert actual_report.shape == (5, 2)
    # Check if it is sorted by importance
    assert actual_report.iloc[0].name == "f4"

    # Check report values
    assert actual_report.loc["f4"]["mean_abs_shap_value"] > 0

    actual_shap_values_test = rm.get_shap_values()
    # 50 test samples and 5 features
    assert actual_shap_values_test.shape == (X1.shape[0], X1.shape[1])

    # Run plots
    rm.plot(plot_type="bar", show=True)
    rm.plot(plot_type="dot", show=False)


def test_permutation_resemblance_class(X1, X2):
    """
    Test.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = PermutationImportanceResemblance(clf, test_prc=0.5, n_jobs=1, random_state=42, iterations=20)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        rm._check_if_fitted()

    actual_report, train_score, test_score = rm.fit_compute(X1, X2, return_scores=True)

    # After the fit this should not raise any error
    rm._check_if_fitted()

    assert train_score == 1
    assert test_score == 1

    # Check report shape
    assert actual_report.shape == (3, 2)
    # Check if it is sorted by importance
    assert actual_report.iloc[0].name == "col_1"
    # Check report values
    assert actual_report.loc["col_1"]["mean_importance"] > 0
    assert actual_report.loc["col_1"]["std_importance"] > 0
    assert actual_report.loc["col_2"]["mean_importance"] == 0
    assert actual_report.loc["col_2"]["std_importance"] == 0
    assert actual_report.loc["col_3"]["mean_importance"] == 0
    assert actual_report.loc["col_3"]["std_importance"] == 0

    rm.plot(figsize=(10, 10))
    # Check plot size
    fig = plt.gcf()
    size = fig.get_size_inches()
    assert size[0] == 10 and size[1] == 10


def test_base_class_same_data(X1):
    """
    Test.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    rm = BaseResemblanceModel(clf, test_prc=0.5, n_jobs=1, random_state=42)

    actual_report, train_score, test_score = rm.fit_compute(X1, X1, return_scores=True)

    assert train_score == 0.5
    assert test_score == 0.5
    assert actual_report is None
