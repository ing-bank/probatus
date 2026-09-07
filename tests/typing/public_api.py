"""Static API contract checks, run by mypy as part of scripts/check.py."""

from collections.abc import Hashable
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import assert_type

from probatus.feature_elimination import EarlyStoppingShapRFECV, ShapRFECV
from probatus.interpret import DependencePlotter, ShapModelInterpreter
from probatus.sample_similarity import PermutationImportanceResemblance, SHAPImportanceResemblance
from probatus.utils import Scorer, assure_pandas_df, preprocess_data, shap_calc, shap_to_df


def check_public_api(model: Any, X: pd.DataFrame, y: pd.Series[Any], flag: bool) -> None:
    """Check concrete defaults, flag-dependent returns, and rejected arguments."""
    names: list[str] = ["first", "second"]
    selector = ShapRFECV(model, scoring=Scorer("roc_auc"))
    assert_type(selector.fit(X, y, column_names=names), ShapRFECV)
    assert_type(selector.compute(), pd.DataFrame)
    assert_type(selector.fit_compute(X, y), pd.DataFrame)
    assert_type(selector.plot(show=False), Figure)
    assert_type(selector.get_reduced_features_set(1), list[Hashable])
    assert_type(selector.get_reduced_features_set("best", return_type="support"), list[bool])
    assert_type(selector.get_reduced_features_set(1, return_type="ranking"), list[int])
    assert_type(EarlyStoppingShapRFECV(model).compute(), pd.DataFrame)

    dependence = DependencePlotter(model)
    assert_type(dependence.fit(X, y), DependencePlotter)
    assert_type(dependence.fit_compute(X, y), pd.DataFrame)
    assert_type(dependence.plot("first", show=False), list[Axes])

    interpreter = ShapModelInterpreter(model)
    assert_type(interpreter.fit(X, X, y, y, column_names=names), None)
    assert_type(interpreter.compute(), pd.DataFrame)
    assert_type(interpreter.compute(return_scores=True), tuple[pd.DataFrame, float, float])
    assert_type(interpreter.compute(return_scores=flag), pd.DataFrame | tuple[pd.DataFrame, float, float])
    assert_type(interpreter.fit_compute(X, X, y, y), pd.DataFrame)
    assert_type(interpreter.fit_compute(X, X, y, y, return_scores=True), tuple[pd.DataFrame, float, float])

    permutation = PermutationImportanceResemblance(model)
    assert_type(permutation.fit(X, X), PermutationImportanceResemblance)
    assert_type(permutation.compute(), pd.DataFrame)
    assert_type(permutation.fit_compute(X, X, return_scores=True), tuple[pd.DataFrame, float, float])
    resemblance = SHAPImportanceResemblance(model)
    assert_type(resemblance.fit_compute(X, X), pd.DataFrame)
    assert_type(resemblance.compute(return_scores=True), tuple[pd.DataFrame, float, float])
    assert_type(resemblance.get_shap_values(), NDArray[Any])

    assert_type(assure_pandas_df([[1, 2]], column_names=names), pd.DataFrame)
    assert_type(preprocess_data(X), tuple[pd.DataFrame, list[Hashable]])
    assert_type(shap_calc(model, X), NDArray[Any])
    assert_type(shap_calc(model, X, return_explainer=True), tuple[NDArray[Any], Any])
    assert_type(shap_to_df(model, X), pd.DataFrame)

    # Unused-ignore checking ensures these mistakes continue to be rejected.
    ShapRFECV(model, cv="invalid")  # type: ignore[arg-type]
    ShapRFECV(model, step="one")  # type: ignore[arg-type]
    selector.fit(X, "invalid labels")  # type: ignore[arg-type]
    interpreter.compute(return_scores="yes")  # type: ignore[call-overload]
