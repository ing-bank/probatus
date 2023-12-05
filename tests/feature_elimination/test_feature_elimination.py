import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from probatus.feature_elimination import EarlyStoppingShapRFECV, ShapRFECV
from probatus.utils import preprocess_labels


@pytest.fixture(scope="function")
def X():
    """
    Fixture for X.
    """
    return pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1, 1, 1, 1, 0],
            "col_2": [0, 0, 0, 0, 0, 0, 0, 1],
            "col_3": [1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )


@pytest.fixture(scope="session")
def catboost_classifier_class():
    """This fixture allows to reuse the import of the CatboostClassifier class across different tests.

    It is equivalent to importing the package at the beginning of the file.

    Importing catboost multiple times results in a ValueError: I/O operation on closed file.

    """
    from catboost import CatBoostClassifier

    return CatBoostClassifier


@pytest.fixture(scope="function")
def y():
    """
    Fixture for y.
    """
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def X_multi():
    """
    Fixture for multi-class X.
    """
    return pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1, 0, 1, 0, 0],
            "col_2": [0, 0, 0, 0, 1, 0, 1, 1],
            "col_3": [1, 0, 2, 0, 2, 0, 1, 0],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )


@pytest.fixture(scope="function")
def y_multi():
    """
    Fixture for multi-class y.
    """
    return pd.Series([1, 0, 2, 0, 2, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def y_reg():
    """
    Fixture for y.
    """
    return pd.Series([100, 1, 101, 2, 99, -1, 102, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def sample_weight():
    """
    Fixture for sample_weight.
    """
    return pd.Series([1, 1, 1, 1, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def groups():
    """
    Fixture for groups.
    """
    return pd.Series(["grp1", "grp1", "grp1", "grp1", "grp2", "grp2", "grp2", "grp2"], index=[1, 2, 3, 4, 5, 6, 7, 8])


def test_shap_rfe_randomized_search(X, y):
    """
    Test with RandomizedSearchCV.
    """
    clf = DecisionTreeClassifier(max_depth=1)
    param_grid = {"criterion": ["gini"], "min_samples_split": [1, 2]}
    search = RandomizedSearchCV(clf, param_grid, cv=2, n_iter=2)
    shap_elimination = ShapRFECV(search, step=0.8, cv=2, scoring="roc_auc", n_jobs=4, random_state=1)
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == 2
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]

    _ = shap_elimination.plot(show=False)


def test_shap_rfe_multi_class(X, y):
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)

    shap_elimination = ShapRFECV(
        clf,
        cv=2,
        scoring="roc_auc_ovr",
        random_state=1,
    )

    report = shap_elimination.fit_compute(X, y, approximate=False, check_additivity=False)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe(X, y, sample_weight):
    """
    Test with ShapRFECV.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    shap_elimination = ShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=2,
        scoring="roc_auc",
        n_jobs=4,
    )
    report = shap_elimination.fit_compute(X, y, sample_weight=sample_weight, approximate=True, check_additivity=False)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe_group_cv(X, y, groups, sample_weight):
    """
    Test ShapRFECV with StratifiedGroupKFold.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=1)
    shap_elimination = ShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=cv,
        scoring="roc_auc",
        n_jobs=4,
    )
    report = shap_elimination.fit_compute(
        X, y, groups=groups, sample_weight=sample_weight, approximate=True, check_additivity=False
    )

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_pipeline_error(X, y):
    """
    Test with ShapRFECV for pipelines.
    """
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("dt", DecisionTreeClassifier(max_depth=1, random_state=1)),
        ]
    )
    with pytest.raises(TypeError):
        shap_elimination = ShapRFECV(
            clf,
            random_state=1,
            step=1,
            cv=2,
            scoring="roc_auc",
            n_jobs=4,
        )
        shap_elimination = shap_elimination.fit(X, y, approximate=True, check_additivity=False)


def test_shap_rfe_linear_model(X, y):
    """
    Test ShapRFECV with linear model.
    """
    clf = LogisticRegression(C=1, random_state=1)
    shap_elimination = ShapRFECV(clf, random_state=1, step=1, cv=2, scoring="roc_auc", n_jobs=4)
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe_svm(X, y):
    """
    Test with ShapRFECV with SVM.
    """
    clf = SVC(C=1, kernel="linear", probability=True)
    shap_elimination = ShapRFECV(clf, random_state=1, step=1, cv=2, scoring="roc_auc", n_jobs=4)
    shap_elimination = shap_elimination.fit(X, y)
    report = shap_elimination.compute()

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe_cols_to_keep(X, y):
    """
    Test for shap_rfe_cv with features to keep parameter.
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    shap_elimination = ShapRFECV(
        clf,
        random_state=1,
        step=2,
        cv=2,
        scoring="roc_auc",
        n_jobs=4,
        min_features_to_select=1,
    )
    report = shap_elimination.fit_compute(X, y, columns_to_keep=["col_2", "col_3"])

    assert report.shape[0] == 2
    reduced_feature_set = set(shap_elimination.get_reduced_features_set(num_features=2))
    assert reduced_feature_set == {"col_2", "col_3"}


def test_shap_rfe_randomized_search_cols_to_keep(X, y):
    """
    Test with ShapRFECV with column to keep param.
    """
    clf = DecisionTreeClassifier(max_depth=1)
    param_grid = {"criterion": ["gini"], "min_samples_split": [1, 2]}
    search = RandomizedSearchCV(clf, param_grid, cv=2, n_iter=2)
    shap_elimination = ShapRFECV(search, step=0.8, cv=2, scoring="roc_auc", n_jobs=4, random_state=1)
    report = shap_elimination.fit_compute(X, y, columns_to_keep=["col_2", "col_3"])

    assert report.shape[0] == 2
    reduced_feature_set = set(shap_elimination.get_reduced_features_set(num_features=2))
    assert reduced_feature_set == {"col_2", "col_3"}


def test_calculate_number_of_features_to_remove():
    """
    Test with ShapRFECV with n features to remove.
    """
    assert 3 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=10, num_features_to_remove=3, min_num_features_to_keep=5
    )
    assert 3 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=8, num_features_to_remove=5, min_num_features_to_keep=5
    )
    assert 0 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=5, num_features_to_remove=1, min_num_features_to_keep=5
    )
    assert 4 == ShapRFECV._calculate_number_of_features_to_remove(
        current_num_of_features=5, num_features_to_remove=7, min_num_features_to_keep=1
    )


def test_shap_automatic_num_feature_selection():
    """
    Test automatic num feature selection methods
    """
    X = pd.DataFrame(
        {
            "col_1": [1, 0, 1, 0, 1, 0, 1, 0],
            "col_2": [0, 0, 0, 0, 0, 1, 1, 1],
            "col_3": [1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    shap_elimination = ShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=2,
        scoring="roc_auc",
        n_jobs=1,
    )
    _ = shap_elimination.fit_compute(X, y, approximate=True, check_additivity=False)

    best_features = shap_elimination.get_reduced_features_set(num_features="best")
    best_coherent_features = shap_elimination.get_reduced_features_set(
        num_features="best_coherent",
    )
    best_parsimonious_features = shap_elimination.get_reduced_features_set(num_features="best_parsimonious")

    assert best_features == ["col_2"]
    assert best_coherent_features == ["col_1", "col_2", "col_3"]
    assert best_parsimonious_features == ["col_2"]


def test_get_feature_shap_values_per_fold(X, y):
    """
    Test with ShapRFECV with features per fold.
    """
    clf = DecisionTreeClassifier(max_depth=1)
    shap_elimination = ShapRFECV(clf)
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold(
        X,
        y,
        clf,
        train_index=[2, 3, 4, 5, 6, 7],
        val_index=[0, 1],
        scorer=get_scorer("roc_auc"),
    )
    assert test_score == 1
    assert train_score > 0.9
    assert shap_values.shape == (2, 3)


def test_shap_rfe_same_features_are_kept_after_each_run():
    """
    Test a use case which appears to be flickering with Probatus 1.8.9 and lower.

    Expected result: every run the same outcome.
    Probatus <= 1.8.9: A different order every time.
    """
    SEED = 1234

    feature_names = [(f"f{num}") for num in range(1, 21)]

    # Code from tutorial on probatus documentation
    X, y = make_classification(
        n_samples=100,
        class_sep=0.05,
        n_informative=6,
        n_features=20,
        random_state=SEED,
        n_redundant=10,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X, columns=feature_names)

    random_forest = RandomForestClassifier(
        random_state=SEED,
        n_estimators=70,
        max_features="log2",
        criterion="entropy",
        class_weight="balanced",
    )

    shap_elimination = ShapRFECV(
        random_forest,
        step=0.2,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
        random_state=SEED,
    )

    report = shap_elimination.fit_compute(X, y, check_additivity=True, seed=SEED)
    # Return the set of features with the best validation accuracy

    kept_features = list(report.iloc[[report["val_metric_mean"].idxmax() - 1]]["features_set"].to_list()[0])

    # Results from the first run
    assert ["f2", "f3", "f6", "f10", "f11", "f12", "f13", "f14", "f15", "f17", "f18", "f19", "f20"] == kept_features


def test_shap_rfe_penalty_factor(X, y):
    """
    Test ShapRFECV with shap_variance_penalty_factor
    """
    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    shap_elimination = ShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=2,
        scoring="roc_auc",
        n_jobs=1,
    )
    report = shap_elimination.fit_compute(
        X, y, shap_variance_penalty_factor=1.0, approximate=True, check_additivity=False
    )

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_1"]


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_complex_dataset(complex_data, complex_lightgbm):
    """
    Test on complex dataset.
    """
    X, y = complex_data

    param_grid = {
        "n_estimators": [5, 7, 10],
        "num_leaves": [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(complex_lightgbm, param_grid, n_iter=1)

    shap_elimination = ShapRFECV(clf=search, step=1, cv=10, scoring="roc_auc", n_jobs=3, verbose=50)

    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == X.shape[1]


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_shap_rfe_early_stopping_lightGBM(complex_data):
    """
    Test EarlyStoppingShapRFECV with a LGBMClassifier.
    """
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(n_estimators=200, max_depth=3)
    X, y = complex_data

    shap_elimination = EarlyStoppingShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=10,
        scoring="roc_auc",
        n_jobs=4,
        early_stopping_rounds=5,
        eval_metric="auc",
    )
    report = shap_elimination.fit_compute(X, y, approximate=False, check_additivity=False)

    assert report.shape[0] == 5
    assert shap_elimination.get_reduced_features_set(1) == ["f5"]


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_shap_rfe_early_stopping_XGBoost(complex_data):
    """
    Test EarlyStoppingShapRFECV with a LGBMClassifier.
    """
    from xgboost import XGBClassifier

    clf = XGBClassifier(n_estimators=200, max_depth=3, random_state=42)
    X, y = complex_data
    X["f1_categorical"] = X["f1_categorical"].astype(float)

    shap_elimination = EarlyStoppingShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=10,
        scoring="roc_auc",
        n_jobs=4,
        early_stopping_rounds=5,
        eval_metric="auc",
    )
    report = shap_elimination.fit_compute(X, y, approximate=False, check_additivity=False)

    assert report.shape[0] == 5
    assert shap_elimination.get_reduced_features_set(1) == ["f4"]


# For now this test fails, catboost has issues with categorical variables and
@pytest.mark.xfail
@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_shap_rfe_early_stopping_CatBoost(complex_data, catboost_classifier_class):
    """
    Test EarlyStoppingShapRFECV with a CatBoostClassifier.
    """
    clf = catboost_classifier_class(random_seed=42)
    X, y = complex_data

    shap_elimination = EarlyStoppingShapRFECV(
        clf,
        random_state=1,
        step=1,
        cv=10,
        scoring="roc_auc",
        n_jobs=4,
        early_stopping_rounds=5,
        eval_metric="auc",
    )
    report = shap_elimination.fit_compute(X, y, approximate=False, check_additivity=False)

    assert report.shape[0] == 5
    assert shap_elimination.get_reduced_features_set(1)[0] in ["f4", "f5"]


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_shap_rfe_randomized_search_early_stopping_lightGBM(complex_data):
    """
    Test EarlyStoppingShapRFECV with RandomizedSearchCV and a LGBMClassifier on complex dataset.
    """
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(n_estimators=200)
    X, y = complex_data

    param_grid = {
        "max_depth": [3, 4, 5],
    }
    search = RandomizedSearchCV(clf, param_grid, cv=2, n_iter=2)
    shap_elimination = EarlyStoppingShapRFECV(
        search,
        step=1,
        cv=10,
        scoring="roc_auc",
        early_stopping_rounds=5,
        eval_metric="auc",
        n_jobs=4,
        verbose=50,
        random_state=1,
    )
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == X.shape[1]
    assert shap_elimination.get_reduced_features_set(1) == ["f5"]

    _ = shap_elimination.plot(show=False)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_get_feature_shap_values_per_fold_early_stopping_lightGBM(complex_data):
    """
    Test with ShapRFECV with features per fold.
    """
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(n_estimators=200, max_depth=3)
    X, y = complex_data
    y = preprocess_labels(y, y_name="y", index=X.index)

    shap_elimination = EarlyStoppingShapRFECV(clf, early_stopping_rounds=5)
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold(
        X,
        y,
        clf,
        train_index=list(range(5, 50)),
        val_index=[0, 1, 2, 3, 4],
        scorer=get_scorer("roc_auc"),
    )
    assert test_score > 0.6
    assert train_score > 0.6
    assert shap_values.shape == (5, 5)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_get_feature_shap_values_per_fold_early_stopping_CatBoost(complex_data, catboost_classifier_class):
    """
    Test with ShapRFECV with features per fold.
    """
    clf = catboost_classifier_class(random_seed=42)
    X, y = complex_data
    X["f1_categorical"] = X["f1_categorical"].astype(str).astype("category")
    y = preprocess_labels(y, y_name="y", index=X.index)

    shap_elimination = EarlyStoppingShapRFECV(clf, early_stopping_rounds=5)
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold(
        X,
        y,
        clf,
        train_index=list(range(5, 50)),
        val_index=[0, 1, 2, 3, 4],
        scorer=get_scorer("roc_auc"),
    )
    assert test_score > 0
    assert train_score > 0.6
    assert shap_values.shape == (5, 5)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_get_feature_shap_values_per_fold_early_stopping_XGBoost(complex_data):
    """
    Test with ShapRFECV with features per fold.
    """
    from xgboost import XGBClassifier

    clf = XGBClassifier(n_estimators=200, max_depth=3, random_state=42)
    X, y = complex_data
    X["f1_categorical"] = X["f1_categorical"].astype(float)
    y = preprocess_labels(y, y_name="y", index=X.index)

    shap_elimination = EarlyStoppingShapRFECV(clf, early_stopping_rounds=5)
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold(
        X,
        y,
        clf,
        train_index=list(range(5, 50)),
        val_index=[0, 1, 2, 3, 4],
        scorer=get_scorer("roc_auc"),
    )
    assert test_score > 0
    assert train_score > 0.6
    assert shap_values.shape == (5, 5)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_EarlyStoppingShapRFECV_no_categorical(complex_data):
    """Test EarlyStoppingShapRFECV when no categorical features are present."""
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(n_estimators=50, max_depth=3, num_leaves=3)

    shap_elimination = EarlyStoppingShapRFECV(
        clf=clf,
        step=0.33,
        cv=5,
        scoring="accuracy",
        eval_metric="logloss",
        early_stopping_rounds=5,
    )
    X, y = complex_data
    X = X.drop(columns=["f1_categorical"])
    report = shap_elimination.fit_compute(X, y, feature_perturbation="tree_path_dependent")

    assert report.shape[0] == X.shape[1]
    assert shap_elimination.get_reduced_features_set(1) == ["f5"]

    _ = shap_elimination.plot(show=False)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_LightGBM_stratified_kfold():
    """
    Test added to check for https://github.com/ing-bank/probatus/issues/170.
    """
    from lightgbm import LGBMClassifier

    X = pd.DataFrame(
        [
            [1, 2, 3, 4, 5, 101, 102, 103, 104, 105],
            [-1, -2, 2, -5, -7, 1, 2, 5, -1, 3],
            ["a", "b"] * 5,  # noisy categorical will dropped first
        ]
    ).transpose()
    X[2] = X[2].astype("category")
    X[1] = X[1].astype("float")
    X[0] = X[0].astype("float")
    y = [0] * 5 + [1] * 5

    clf = LGBMClassifier()
    n_iter = 2
    n_folds = 3

    for _ in range(n_iter):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)
        shap_elimination = EarlyStoppingShapRFECV(
            clf=clf,
            step=1 / (n_iter + 1),
            cv=skf,
            scoring="accuracy",
            eval_metric="logloss",
            early_stopping_rounds=5,
        )
        report = shap_elimination.fit_compute(X, y, feature_perturbation="tree_path_dependent")

    assert report.shape[0] == X.shape[1]

    shap_elimination.plot(show=False)
