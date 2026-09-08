"""Exercise the supported array and dataframe inputs to SHAP helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from probatus.utils import assure_pandas_series, calculate_shap_importance, shap_calc, shap_to_df


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
