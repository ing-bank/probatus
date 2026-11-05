"""Exercise the supported array and dataframe inputs to SHAP helpers."""

import numpy as np
import pandas as pd
import pytest
import shap.explainers._tree as shap_tree
from lightgbm import LGBMClassifier
from shap import Explainer
from shap.utils import sample
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from probatus.utils import assure_pandas_series, calculate_shap_importance, shap_calc, shap_to_df
from probatus.utils.shap_helpers import _tree_shap_values


def test_series_without_replacement_index():
    labels = pd.Series([0, 1], index=["first", "second"])
    assert assure_pandas_series(labels) is labels


def test_series_reordered_to_requested_index():
    labels = pd.Series([0, 1], index=["first", "second"])
    result = assure_pandas_series(labels, index=["second", "first"])
    pd.testing.assert_series_equal(result, pd.Series([1, 0], index=["second", "first"]))


@pytest.mark.parametrize("as_frame", [False, True])
def test_shap_array_and_dataframe_inputs(as_frame):
    values = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    X = pd.DataFrame(values, columns=["a", "b"], index=[10, 20, 30, 40]) if as_frame else values
    model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, [0, 0, 1, 1])
    shap_values, explainer = shap_calc(model, X, return_explainer=True, random_state=42)
    assert shap_values.shape == (4, 2)
    assert hasattr(explainer, "expected_value")
    frame = shap_to_df(model, X, precalc_shap=shap_values)
    assert frame.columns.tolist() == (["a", "b"] if as_frame else ["col_0", "col_1"])
    assert frame.index.tolist() == ([10, 20, 30, 40] if as_frame else [0, 1, 2, 3])
    np.testing.assert_allclose(frame.to_numpy(), shap_values)
    pd.testing.assert_frame_equal(shap_to_df(model, X, random_state=42), frame)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize("multiclass", [False, True])
@pytest.mark.parametrize("penalty", [None, 0, -1, 0.5, 1])
def test_shap_importance_matches_existing_aggregation(dtype, layout, multiclass, penalty):
    """Keep values, tie ordering, duplicate labels and dtypes identical to the original calculation."""
    rng = np.random.default_rng(42)
    shape = (3, 31, 24) if multiclass else (31, 24)
    values = rng.normal(size=shape).astype(dtype)
    values[..., 0] = values[..., 2]  # Tied importance, including on strided inputs.
    values[..., 4] = 0
    values[..., 6] = 0
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[..., ::2]
    original = values.copy()
    values.flags.writeable = False
    columns = [0, "duplicate", "duplicate", *range(3, values.shape[-1])]

    # Reference the previous public output, including pandas' default sorting of ties.
    absolute = np.abs(values)
    signed = values
    if multiclass:
        absolute = absolute.sum(axis=0)
        signed = signed.sum(axis=0)
    mean_absolute = absolute.mean(axis=0)
    expected = pd.DataFrame(
        {
            "mean_abs_shap_value_test": mean_absolute,
            "mean_shap_value_test": signed.mean(axis=0),
            "penalized": mean_absolute - absolute.std(axis=0) * max(penalty or 0, 0),
        },
        index=columns,
    ).astype(float)
    expected = expected.sort_values("penalized", ascending=False).drop(columns="penalized")

    result = calculate_shap_importance(values, columns, "_test", penalty)

    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("penalty", [None, 0, -1])
def test_disabled_shap_penalty_does_not_overflow_variance(penalty):
    """A disabled penalty must not turn large finite importance into a NaN sort key."""
    values = np.array([[1e200, 1.0], [0.0, 1.0]])
    with np.errstate(over="raise", invalid="raise"):
        result = calculate_shap_importance(values, ["large", "small"], shap_variance_penalty_factor=penalty)
    expected = pd.DataFrame(
        {"mean_abs_shap_value": [5e199, 1.0], "mean_shap_value": [5e199, 1.0]}, index=["large", "small"]
    )
    pd.testing.assert_frame_equal(result, expected)


def test_shap_variance_penalty_changes_ranking_but_not_reported_means():
    values = np.array([[0.0, 2.0], [6.0, 2.0]])
    unpenalized = calculate_shap_importance(values, ["variable", "constant"])
    penalized = calculate_shap_importance(values, ["variable", "constant"], shap_variance_penalty_factor=1)
    assert unpenalized.index.tolist() == ["variable", "constant"]
    assert penalized.index.tolist() == ["constant", "variable"]
    pd.testing.assert_frame_equal(penalized, unpenalized.iloc[::-1])


@pytest.mark.parametrize("penalty", [None, 0.5])
def test_shap_importance_sorts_nan_features_last(penalty):
    values = np.array([[np.nan, 0.0, 1.0], [1.0, 0.0, 1.0]])
    result = calculate_shap_importance(values, ["missing", "zero", "one"], shap_variance_penalty_factor=penalty)
    assert result.index.tolist() == ["one", "zero", "missing"]
    assert result.loc["missing"].isna().all()


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


@pytest.mark.parametrize("perturbation", [{}, {"feature_perturbation": "tree_path_dependent"}])
def test_tree_model_is_parsed_once(monkeypatch, classification_data, perturbation):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0).fit(X, y)
    parses = []
    original_init = shap_tree.TreeEnsemble.__init__

    def counting_init(self, *args, **kwargs):
        parses.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(shap_tree.TreeEnsemble, "__init__", counting_init)
    shap_calc(model, X, random_state=0, **perturbation)
    assert len(parses) == 1


@pytest.mark.parametrize(
    "model", [DecisionTreeClassifier(max_depth=3, random_state=0), LogisticRegression(), LinearRegression()]
)
def test_explainer_matches_shap_automatic_selection(classification_data, model):
    X, y = classification_data
    model.fit(X, y)
    _, explainer = shap_calc(model, X, return_explainer=True, random_state=0)
    reference = Explainer(model, masker=sample(X, 100, random_state=0), seed=0)
    assert type(explainer) is type(reference)


@pytest.mark.parametrize("perturbation", [{}, {"feature_perturbation": "tree_path_dependent"}])
def test_unsupported_model_keeps_shap_error(classification_data, perturbation):
    X, y = classification_data
    model = KNeighborsClassifier().fit(X, y)
    with pytest.raises(TypeError, match="not callable"):
        shap_calc(model, X, random_state=0, **perturbation)


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize(
    "model",
    [
        RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0),
        LGBMClassifier(n_estimators=10, verbose=-1, random_state=0),
    ],
)
def test_parallel_shap_values_match_single_process(classification_data, model, as_frame):
    X, y = classification_data
    X = X if as_frame else X.to_numpy()
    model.fit(X, y)
    serial, serial_explainer = shap_calc(model, X, return_explainer=True, random_state=0, n_jobs=1)
    parallel, parallel_explainer = shap_calc(model, X, return_explainer=True, random_state=0, n_jobs=2)
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(parallel_explainer.expected_value, serial_explainer.expected_value)


class RecordingExplainer:
    """Stands in for an interventional tree explainer; echoes each row and the additivity flag it received."""

    feature_perturbation = "interventional"

    def shap_values(self, X, check_additivity=True, approximate=False):
        return np.column_stack([X[:, 0], np.full(len(X), check_additivity)])


@pytest.mark.parametrize("check_additivity", [False, True])
def test_parallel_chunks_keep_row_order_and_options(check_additivity):
    X = np.arange(11, dtype=float).reshape(-1, 1)
    values = _tree_shap_values(
        RecordingExplainer(), X, check_additivity=check_additivity, approximate=False, verbose=0, n_jobs=3
    )
    np.testing.assert_array_equal(values[:, 0], X[:, 0])
    assert (values[:, 1] == check_additivity).all()
