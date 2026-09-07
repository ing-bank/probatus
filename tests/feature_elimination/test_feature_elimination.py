import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from probatus.feature_elimination import EarlyStoppingShapRFECV, ShapRFECV
from probatus.utils import preprocess_labels


@pytest.fixture(scope="function")
def X():
    return pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1, 1, 1, 1, 0],
            "col_2": [0, 0, 0, 0, 0, 0, 0, 1],
            "col_3": [1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )


@pytest.fixture(scope="function")
def y():
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def sample_weight():
    return pd.Series([1, 1, 1, 1, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def groups():
    return pd.Series(["grp1", "grp1", "grp1", "grp1", "grp2", "grp2", "grp2", "grp2"], index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture(scope="function")
def XGBoost_classifier(random_state):
    model = XGBClassifier(n_estimators=200, max_depth=3, random_state=random_state)
    return model


@pytest.fixture(scope="function")
def XGBoost_regressor(random_state):
    model = XGBRegressor(n_estimators=200, max_depth=3, random_state=random_state)
    return model


def test_shap_rfe_regressor(XGBoost_regressor, random_state):
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target

    shap_elimination = ShapRFECV(XGBoost_regressor, step=0.8, cv=2, scoring="r2", n_jobs=4, random_state=random_state)
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["bmi"]

    _ = shap_elimination.plot(show=False)


def test_shap_rfe_randomized_search(X, y, randomized_search_decision_tree_classifier, random_state):
    search = randomized_search_decision_tree_classifier
    shap_elimination = ShapRFECV(search, step=0.8, cv=2, scoring="roc_auc", n_jobs=4, random_state=random_state)
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == 2
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]

    _ = shap_elimination.plot(show=False)


def test_shap_rfe_multi_class(X, y, decision_tree_classifier, random_state):
    shap_elimination = ShapRFECV(
        decision_tree_classifier,
        cv=2,
        scoring="roc_auc_ovr",
        random_state=random_state,
    )

    report = shap_elimination.fit_compute(X, y, approximate=False, check_additivity=False)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe(X, y, sample_weight, decision_tree_classifier, random_state):
    shap_elimination = ShapRFECV(
        decision_tree_classifier,
        random_state=random_state,
        step=1,
        cv=2,
        scoring="roc_auc",
        n_jobs=4,
    )
    report = shap_elimination.fit_compute(X, y, sample_weight=sample_weight, approximate=True, check_additivity=False)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe_group_cv(X, y, groups, sample_weight, decision_tree_classifier, random_state):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    shap_elimination = ShapRFECV(
        decision_tree_classifier,
        random_state=random_state,
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


def test_shap_pipeline_error(X, y, decision_tree_classifier, random_state):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("dt", decision_tree_classifier),
        ]
    )
    with pytest.raises(TypeError):
        shap_elimination = ShapRFECV(
            model,
            random_state=random_state,
            step=1,
            cv=2,
            scoring="roc_auc",
            n_jobs=4,
        )
        shap_elimination = shap_elimination.fit(X, y, approximate=True, check_additivity=False)


def test_shap_rfe_linear_model(X, y, random_state):
    model = LogisticRegression(C=1, random_state=random_state)
    shap_elimination = ShapRFECV(model, random_state=random_state, step=1, cv=2, scoring="roc_auc", n_jobs=4)
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe_svm(X, y, random_state):
    model = SVC(C=1, kernel="linear", probability=True, random_state=random_state)
    shap_elimination = ShapRFECV(model, random_state=random_state, step=1, cv=2, scoring="roc_auc", n_jobs=4)
    shap_elimination = shap_elimination.fit(X, y)
    report = shap_elimination.compute()

    assert report.shape[0] == 3
    assert shap_elimination.get_reduced_features_set(1) == ["col_3"]


def test_shap_rfe_cols_to_keep(X, y, decision_tree_classifier, random_state):
    shap_elimination = ShapRFECV(
        decision_tree_classifier,
        random_state=random_state,
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


def test_shap_rfe_randomized_search_cols_to_keep(X, y, randomized_search_decision_tree_classifier, random_state):
    search = randomized_search_decision_tree_classifier
    shap_elimination = ShapRFECV(search, step=0.8, cv=2, scoring="roc_auc", n_jobs=4, random_state=random_state)
    report = shap_elimination.fit_compute(X, y, columns_to_keep=["col_2", "col_3"])

    assert report.shape[0] == 2
    reduced_feature_set = set(shap_elimination.get_reduced_features_set(num_features=2))
    assert reduced_feature_set == {"col_2", "col_3"}


def test_calculate_number_of_features_to_remove():
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


@pytest.mark.parametrize(
    "method, threshold, expected_features",
    [
        ("best", 0.75, ["signal", "stable", "extra"]),
        ("best_coherent", 0.75, ["signal", "stable", "extra", "noise"]),
        ("best_parsimonious", 0.75, ["signal", "stable"]),
        ("best", 0.0, ["signal", "stable", "extra"]),
        ("best_coherent", 0.0, ["signal", "stable", "extra"]),
        ("best_parsimonious", 0.0, ["signal", "stable", "extra"]),
    ],
)
def test_shap_automatic_num_feature_selection(decision_tree_classifier, method, threshold, expected_features):
    # Test selection policies against a fixed CV report, independently of model/SHAP version changes.
    # Three features score best; four are most stable within 0.75 * 0.04; two are the smallest set within that range.
    # The one-feature result is even more stable, but its score is outside the threshold.
    shap_elimination = ShapRFECV(decision_tree_classifier)
    shap_elimination.report_df = pd.DataFrame(
        {
            "num_features": [4, 3, 2, 1],
            "features_set": [
                ["signal", "stable", "extra", "noise"],
                ["signal", "stable", "extra"],
                ["signal", "stable"],
                ["signal"],
            ],
            "val_metric_mean": [0.89, 0.90, 0.88, 0.60],
            "val_metric_std": [0.01, 0.04, 0.02, 0.0],
        },
        index=[1, 2, 3, 4],
    )
    shap_elimination.fitted = True

    assert (
        shap_elimination.get_reduced_features_set(num_features=method, standard_error_threshold=threshold)
        == expected_features
    )


def test_get_feature_shap_values_per_fold(X, y, decision_tree_classifier, random_state):
    shap_elimination = ShapRFECV(decision_tree_classifier, scoring="roc_auc", random_state=random_state)
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold(
        X,
        y,
        decision_tree_classifier,
        train_index=[2, 3, 4, 5, 6, 7],
        val_index=[0, 1],
    )
    assert test_score == 1
    assert train_score > 0.9
    assert shap_values.shape == (2, 3)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_shap_rfe_same_features_are_kept_after_each_run(random_state_1234, n_jobs):
    """
    Test a use case which appears to be flickering with Probatus 1.8.9 and lower.

    Expected result: every run the same outcome.
    Probatus <= 1.8.9: A different order every time.
    """
    feature_names = [(f"f{num}") for num in range(1, 21)]

    # Code from tutorial on probatus documentation
    X, y = make_classification(
        n_samples=100,
        class_sep=0.05,
        n_informative=6,
        n_features=20,
        random_state=random_state_1234,
        n_redundant=10,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X, columns=feature_names)

    random_forest = RandomForestClassifier(
        random_state=random_state_1234,
        n_estimators=70,
        max_features="log2",
        criterion="entropy",
        class_weight="balanced",
    )

    reports = []
    selected_features = []
    for _ in range(2):
        shap_elimination = ShapRFECV(
            random_forest,
            step=0.2,
            cv=5,
            scoring="f1_macro",
            n_jobs=n_jobs,
            random_state=random_state_1234,
        )
        reports.append(shap_elimination.fit_compute(X, y, check_additivity=True))
        selected_features.append(shap_elimination.get_reduced_features_set("best"))

    # Reproducibility means identical runs in the installed stack, not matching an old stack's feature list.
    pd.testing.assert_frame_equal(reports[0], reports[1], check_exact=True)
    assert selected_features[0] == selected_features[1]
    assert reports[0]["num_features"].tolist()[0] == len(feature_names)
    assert reports[0]["num_features"].tolist()[-1] == 1
    for features in reports[0]["features_set"]:
        assert features == [feature for feature in feature_names if feature in features]


@pytest.mark.parametrize(
    "penalty_factor, expected_feature, first_eliminated",
    [(None, "col_2", "col_3"), (0.0, "col_2", "col_3"), (1.0, "col_1", "col_2")],
)
def test_shap_rfe_penalty_factor(
    X, y, decision_tree_classifier, random_state, monkeypatch, penalty_factor, expected_feature, first_eliminated
):
    # col_2 has greater mean importance (3) but also greater variability (3) than col_1 (mean 2, std 0).
    # Supplying known SHAP values isolates the penalty's effect while retaining fitting, CV and elimination.
    shap_values = pd.DataFrame({"col_1": [2.0] * 8, "col_2": [0.0, 6.0] * 4, "col_3": [0.5] * 8}, index=X.index)

    def controlled_shap_calc(model, X, **kwargs):
        return shap_values.loc[X.index, X.columns].to_numpy()

    monkeypatch.setattr("probatus.feature_elimination.feature_elimination.shap_calc", controlled_shap_calc)
    shap_elimination = ShapRFECV(
        decision_tree_classifier,
        random_state=random_state,
        step=1,
        cv=2,
        scoring="roc_auc",
        n_jobs=1,
    )
    report = shap_elimination.fit_compute(X, y, shap_variance_penalty_factor=penalty_factor)

    assert report["num_features"].tolist() == [3, 2, 1]
    assert report.iloc[0]["eliminated_features"] == [first_eliminated]
    assert shap_elimination.get_reduced_features_set(1) == [expected_feature]


def test_complex_dataset(complex_data, complex_lightgbm, random_state_1):
    X, y = complex_data

    param_grid = {
        "n_estimators": [5, 7, 10],
        "num_leaves": [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(complex_lightgbm, param_grid, n_iter=1, random_state=random_state_1)

    shap_elimination = ShapRFECV(
        model=search, step=1, cv=10, scoring="roc_auc", n_jobs=3, verbose=1, random_state=random_state_1
    )

    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == X.shape[1]


def test_shap_rfe_early_stopping_lightGBM(complex_data, random_state):
    model = LGBMClassifier(n_estimators=200, max_depth=3, random_state=random_state)
    X, y = complex_data

    shap_elimination = EarlyStoppingShapRFECV(
        model,
        random_state=random_state,
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


def test_shap_rfe_early_stopping_XGBoost(XGBoost_classifier, complex_data, random_state):
    X, y = complex_data
    X["f1_categorical"] = X["f1_categorical"].astype(float)

    shap_elimination = ShapRFECV(
        XGBoost_classifier,
        random_state=random_state,
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


#
#
def test_shap_rfe_early_stopping_CatBoost(complex_data_with_categorical, catboost_classifier, random_state):
    X, y = complex_data_with_categorical

    shap_elimination = ShapRFECV(
        catboost_classifier,
        random_state=random_state,
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


def test_shap_rfe_randomized_search_early_stopping_lightGBM(complex_data, random_state):
    model = LGBMClassifier(n_estimators=200, random_state=random_state)
    X, y = complex_data

    param_grid = {
        "max_depth": [3, 4, 5],
    }
    search = RandomizedSearchCV(model, param_grid, cv=2, n_iter=2, random_state=random_state)
    shap_elimination = ShapRFECV(
        search,
        step=1,
        cv=10,
        scoring="roc_auc",
        early_stopping_rounds=5,
        eval_metric="auc",
        n_jobs=4,
        verbose=1,
        random_state=random_state,
    )
    report = shap_elimination.fit_compute(X, y)

    assert report.shape[0] == X.shape[1]
    assert shap_elimination.get_reduced_features_set(1) == ["f5"]

    _ = shap_elimination.plot(show=False)


def test_get_feature_shap_values_per_fold_early_stopping_lightGBM(complex_data, random_state):
    model = LGBMClassifier(n_estimators=200, max_depth=3, random_state=random_state)
    X, y = complex_data
    y = preprocess_labels(y, y_name="y", index=X.index)

    shap_elimination = ShapRFECV(model, early_stopping_rounds=5, scoring="roc_auc", random_state=random_state)
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold_early_stopping(
        X,
        y,
        model,
        train_index=list(range(5, 50)),
        val_index=[0, 1, 2, 3, 4],
    )
    assert test_score > 0.6
    assert train_score > 0.6
    assert shap_values.shape == (5, 5)


def test_get_feature_shap_values_per_fold_early_stopping_CatBoost(
    complex_data_with_categorical, catboost_classifier, random_state
):
    X, y = complex_data_with_categorical
    y = preprocess_labels(y, y_name="y", index=X.index)

    shap_elimination = ShapRFECV(
        catboost_classifier, early_stopping_rounds=5, scoring="roc_auc", random_state=random_state
    )
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold_early_stopping(
        X,
        y,
        catboost_classifier,
        train_index=list(range(5, 50)),
        val_index=[0, 1, 2, 3, 4],
    )
    assert test_score > 0
    assert train_score > 0.6
    assert shap_values.shape == (5, 5)


def test_get_feature_shap_values_per_fold_early_stopping_XGBoost(XGBoost_classifier, complex_data, random_state):
    X, y = complex_data
    y = preprocess_labels(y, y_name="y", index=X.index)

    shap_elimination = ShapRFECV(
        XGBoost_classifier, early_stopping_rounds=5, scoring="roc_auc", random_state=random_state
    )
    (
        shap_values,
        train_score,
        test_score,
    ) = shap_elimination._get_feature_shap_values_per_fold_early_stopping(
        X,
        y,
        XGBoost_classifier,
        train_index=list(range(5, 50)),
        val_index=[0, 1, 2, 3, 4],
    )
    assert test_score > 0
    assert train_score > 0.6
    assert shap_values.shape == (5, 5)


def test_EarlyStoppingShapRFECV_no_categorical(complex_data, random_state):
    model = LGBMClassifier(n_estimators=50, max_depth=3, num_leaves=3, random_state=random_state)

    shap_elimination = ShapRFECV(
        model=model,
        step=0.33,
        cv=5,
        scoring="accuracy",
        eval_metric="logloss",
        early_stopping_rounds=5,
        random_state=random_state,
    )
    X, y = complex_data
    X = X.drop(columns=["f1_categorical"])
    report = shap_elimination.fit_compute(X, y, feature_perturbation="tree_path_dependent")

    assert report.shape[0] == X.shape[1]
    assert shap_elimination.get_reduced_features_set(1) == ["f5"]

    _ = shap_elimination.plot(show=False)


def test_LightGBM_stratified_kfold(random_state):
    """
    Test added to check for https://github.com/ing-bank/probatus/issues/170.
    """
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

    model = LGBMClassifier(random_state=random_state)
    n_iter = 2
    n_folds = 3

    for _ in range(n_iter):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
        shap_elimination = ShapRFECV(
            model=model,
            step=1 / (n_iter + 1),
            cv=skf,
            scoring="accuracy",
            eval_metric="logloss",
            early_stopping_rounds=5,
            random_state=random_state,
        )
        report = shap_elimination.fit_compute(X, y, feature_perturbation="tree_path_dependent")

    assert report.shape[0] == X.shape[1]

    shap_elimination.plot(show=False)
