"""Exercise the supported array and dataframe inputs to SHAP helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from probatus.utils import assure_pandas_series, shap_calc, shap_to_df


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
