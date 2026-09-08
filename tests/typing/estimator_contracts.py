"""Estimator contracts checked without untyped sklearn imports or inheritance."""

from collections import UserList

import numpy as np
import pandas as pd
from typing_extensions import Self, assert_type

from probatus._typing import Array, Estimator, WeightedEstimator
from probatus.feature_elimination import EarlyStoppingShapRFECV, ShapRFECV
from probatus.interpret import DependencePlotter, ShapModelInterpreter
from probatus.sample_similarity import PermutationImportanceResemblance
from probatus.utils import assure_pandas_df, assure_pandas_series


class TypedEstimator:
    """An independently typed estimator that does not support sample weights."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        return self

    def predict(self, X: pd.DataFrame) -> Array:
        return np.zeros(len(X), dtype=np.int64)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    def set_params(self, **params: object) -> Self:
        return self


class TypedWeightedEstimator(TypedEstimator):
    """Weights may be keyword-only, as in sklearn-compatible estimators."""

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> Self:
        return self


class InvalidWeightedEstimator(TypedEstimator):
    """Valid basic fitting, but an incompatible sample-weight type."""

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: str | None = None) -> Self:
        return self


class CustomSelector(ShapRFECV):
    """A subclass must retain its type after fitting."""


class CustomDependencePlotter(DependencePlotter):
    """A subclass must retain its type after fitting."""


class CustomResemblance(PermutationImportanceResemblance):
    """A subclass must retain its type after fitting."""


def accepts_estimator(model: Estimator, X: pd.DataFrame, y: pd.Series) -> None:
    assert_type(model.fit(X, y), Estimator)
    assert_type(model.set_params(), Estimator)


def accepts_weights(model: WeightedEstimator, X: pd.DataFrame, y: pd.Series) -> None:
    assert_type(model.fit(X, y, sample_weight=y), WeightedEstimator)
    assert_type(model.set_params(), WeightedEstimator)


def check_estimator_contracts(X: pd.DataFrame, y: pd.Series) -> None:
    basic = TypedEstimator()
    weighted = TypedWeightedEstimator()
    accepts_estimator(basic, X, y)
    accepts_estimator(weighted, X, y)
    accepts_weights(weighted, X, y)
    accepts_weights(basic, X, y)  # type: ignore[arg-type]
    accepts_weights(InvalidWeightedEstimator(), X, y)  # type: ignore[arg-type]
    assert_type(basic.fit(X, y).set_params(), TypedEstimator)
    assert_type(weighted.fit(X, y, sample_weight=y).set_params(), TypedWeightedEstimator)

    assert_type(ShapRFECV(basic).fit(X, y), ShapRFECV)
    assert_type(ShapRFECV(weighted).fit(X, y, sample_weight=y), ShapRFECV)
    assert_type(EarlyStoppingShapRFECV(weighted).fit(X, y), EarlyStoppingShapRFECV)
    assert_type(CustomSelector(basic).fit(X, y), CustomSelector)
    assert_type(CustomDependencePlotter(basic).fit(X, y), CustomDependencePlotter)
    assert_type(CustomResemblance(basic).fit(X, X), CustomResemblance)


def check_list_inputs(model: Estimator) -> None:
    rows: list[list[float]] = [[0.1, 0.2], [0.3, 0.4]]
    labels: list[str] = ["a", "b"]
    weights: list[float] = [1.0, 0.5]
    groups: list[int] = [0, 1]
    assert_type(assure_pandas_df(rows), pd.DataFrame)
    assert_type(assure_pandas_series(labels, index=("first", "second")), pd.Series)
    assert_type(ShapRFECV(model).fit(rows, labels, sample_weight=weights, groups=groups), ShapRFECV)
    assert_type(ShapModelInterpreter(model).fit_compute(rows, rows, labels, labels), pd.DataFrame)
    assure_pandas_df("invalid")  # type: ignore[arg-type]
    assure_pandas_df(((0.1, 0.2), (0.3, 0.4)))  # type: ignore[arg-type]
    assure_pandas_df(UserList(rows))  # type: ignore[arg-type]
    assure_pandas_series("invalid")  # type: ignore[arg-type]
    assure_pandas_series((0, 1))  # type: ignore[arg-type]
    assure_pandas_series(UserList(labels))  # type: ignore[arg-type]


def check_independent_input_types(
    model: Estimator, X1: list[list[int]], X2: list[list[float]], y1: list[int], y2: list[float]
) -> None:
    """Paired datasets and labels need not use identical list element types."""
    assert_type(ShapModelInterpreter(model).fit_compute(X1, X2, y1, y2), pd.DataFrame)
    assert_type(PermutationImportanceResemblance(model).fit_compute(X1, X2), pd.DataFrame)
