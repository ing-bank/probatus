"""Behavioral regressions for search integration and feature-selection policies."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn import config_context
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from probatus.feature_elimination import ShapRFECV


class RecordingGroupKFold(GroupKFold):
    """Assert that every inner and outer CV split respects the supplied groups."""

    calls = 0

    def split(self, X, y=None, groups=None):
        assert groups is not None
        type(self).calls += 1
        for train, test in super().split(X, y, groups):
            assert set(np.asarray(groups)[train]).isdisjoint(np.asarray(groups)[test])
            yield train, test


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("search_class", [GridSearchCV, RandomizedSearchCV])
@pytest.mark.parametrize("grouped", [False, True])
def test_search_metadata_routing(routing, search_class, grouped):
    X, y = make_classification(n_samples=48, n_features=4, n_informative=2, random_state=42)
    groups = np.repeat(np.arange(12), 4) if grouped else None
    RecordingGroupKFold.calls = 0
    with config_context(enable_metadata_routing=routing):
        search_kwargs = {"n_iter": 2, "random_state": 42} if search_class is RandomizedSearchCV else {}
        search = search_class(
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [1, 2]},
            cv=RecordingGroupKFold(2) if grouped else 2,
            scoring="accuracy",
            **search_kwargs,
        )
        selector = ShapRFECV(
            search,
            step=2,
            cv=RecordingGroupKFold(2) if grouped else 2,
            scoring="accuracy",
            n_jobs=1,
            random_state=42,
        )
        report = selector.fit_compute(X, y, groups=groups)
    assert report.num_features.tolist() == [4, 2, 1]
    assert np.isfinite(report.val_metric_mean).all()
    assert not hasattr(search, "best_params_")  # The user's search remains unfitted.
    if grouped:
        assert RecordingGroupKFold.calls == 2 * len(report)


@pytest.mark.parametrize("routing", [False, True])
def test_search_with_early_stopping_and_routing(routing):
    X, y = make_classification(n_samples=48, n_features=4, random_state=42)
    with config_context(enable_metadata_routing=routing):
        search = GridSearchCV(LGBMClassifier(n_estimators=5, verbosity=-1, n_jobs=1), {"max_depth": [2]}, cv=2)
        selector = ShapRFECV(
            search,
            step=3,
            cv=2,
            scoring="accuracy",
            n_jobs=1,
            early_stopping_rounds=2,
            eval_metric="binary_logloss",
            random_state=42,
        )
        report = selector.fit_compute(X, y)
    assert report.num_features.tolist() == [4, 1]


@pytest.mark.parametrize("method, expected", [("best", 4), ("best_coherent", 3), ("best_parsimonious", 2)])
@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (100.0, 0.0), (1.0, -2.0)])
def test_selection_uses_best_score_variability(method, expected, scale, offset):
    selector = ShapRFECV(DecisionTreeClassifier())
    selector.report_df = pd.DataFrame(
        {
            "num_features": [4, 3, 2, 1],
            "features_set": [list(range(n)) for n in [4, 3, 2, 1]],
            "val_metric_mean": np.array([0.9, 0.875, 0.85, 0.6]) * scale + offset,
            "val_metric_std": np.array([0.05, 0.01, 0.04, 0.5]) * scale,
        },
        index=[10, 20, 30, 40],
    )
    selector.fitted = True
    # The boundary is based on the BEST row's std, not each candidate's std.
    assert len(selector.get_reduced_features_set(method)) == expected
    assert len(selector.get_reduced_features_set(method, standard_error_threshold=0)) == 4
    selector.report_df.loc[10, "val_metric_std"] = 0.0
    assert len(selector.get_reduced_features_set(method)) == 4


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_lightgbm_early_stopping_respects_verbosity(verbose, capsys):
    X, y = make_classification(n_samples=48, n_features=4, random_state=42)
    selector = ShapRFECV(
        LGBMClassifier(n_estimators=5, verbosity=-1, n_jobs=1),
        step=3,
        cv=2,
        scoring="accuracy",
        n_jobs=1,
        early_stopping_rounds=2,
        eval_metric="binary_logloss",
        verbose=verbose,
        random_state=42,
    )
    selector.fit(X, y)
    output = capsys.readouterr().out
    if verbose < 2:
        assert output == ""
    else:
        assert "Training until validation scores" in output


def test_support_and_ranking_with_named_columns():
    selector = ShapRFECV(DecisionTreeClassifier())
    selector.column_names = ["a", "b", "c"]
    selector.report_df = pd.DataFrame(
        {
            "num_features": [3, 2, 1],
            "features_set": [["a", "b", "c"], ["a", "c"], ["c"]],
            "eliminated_features": [["b"], ["a"], []],
        }
    )
    selector.fitted = True
    assert selector.get_reduced_features_set(2, return_type="support") == [True, False, True]
    # Preserve existing zero-based ranking semantics while accepting string labels.
    assert selector.get_reduced_features_set(2, return_type="ranking") == [1, 2, 0]
    with pytest.raises(ValueError, match="num_features"):
        selector.get_reduced_features_set(2.5)
