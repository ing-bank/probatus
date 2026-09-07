"""Data, estimator, and option contracts shared by the public APIs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from typing import Literal, Protocol, TypeAlias, TypedDict, TypeVar, overload

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from typing_extensions import Unpack

Feature: TypeAlias = Hashable
Array: TypeAlias = NDArray[np.generic]
FloatArray: TypeAlias = NDArray[np.float32 | np.float64]
IndexArray: TypeAlias = NDArray[np.int32 | np.int64]
Scalar: TypeAlias = str | int | float | bool | np.generic | None
_Value = TypeVar("_Value", covariant=True)


class DataSequence(Protocol[_Value]):
    """Read-only list operations; copy excludes bare strings from tabular inputs."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_Value]: ...
    @overload
    def __getitem__(self, index: int, /) -> _Value: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_Value]: ...
    def copy(self) -> Sequence[_Value]: ...


Data: TypeAlias = pd.DataFrame | pd.Series | Array | DataSequence[Scalar | DataSequence[Scalar]]
Labels: TypeAlias = pd.Series | Array | DataSequence[Scalar]
Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]


class Estimator(Protocol):
    """The sklearn-style estimator operations used by probatus."""

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Estimator: ...
    def predict(self, X: pd.DataFrame) -> Array: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> Estimator: ...


ScorerCallable: TypeAlias = Callable[[Estimator, Data, Labels], float]
# Boosting libraries use different callback arguments; the numeric result is shared.
EvalMetric: TypeAlias = str | Callable[..., float | tuple[str, float] | tuple[str, float, bool]]


class ShapExplainer(Protocol):
    """The fitted explainer state returned with SHAP values."""

    @property
    def expected_value(self) -> float | FloatArray | list[float]: ...

    def shap_values(self, X: pd.DataFrame | Array, **kwargs: object) -> FloatArray | list[FloatArray]: ...


class CVSplitter(Protocol):
    """The splitter interface consumed by feature elimination."""

    def split(
        self, X: Data, y: Labels | None = None, groups: Labels | None = None
    ) -> Iterable[tuple[IndexArray, IndexArray]]: ...


CV: TypeAlias = int | CVSplitter | Iterable[tuple[IndexArray, IndexArray]] | None


class ExplainerOptions(TypedDict, total=False):
    """Options forwarded to SHAP's built-in explainers."""

    algorithm: Literal["auto", "permutation", "partition", "tree", "linear", "deep", "exact", "additive"]
    link: Callable[[FloatArray], FloatArray]
    linearize_link: bool
    output_names: list[str]
    feature_names: list[str] | list[list[str]]
    feature_perturbation: Literal["auto", "interventional", "tree_path_dependent", "correlation_dependent"]
    model_output: str
    nsamples: int
    npermutations: int


class ShapOptionsWithoutApproximation(ExplainerOptions, total=False):
    """Options shared by SHAP calculation and interpreter preprocessing."""

    sample_size: int
    check_additivity: bool


class ShapOptions(ShapOptionsWithoutApproximation, total=False):
    """Options accepted when an analyzer computes SHAP values."""

    approximate: bool


class ShapCalculationOptions(ShapOptions, total=False):
    """Calculation options for the standalone dataframe helper."""

    verbose: int
    random_state: int | None


class FigureOptions(TypedDict, total=False):
    """Figure and subplot options accepted by probatus plotting methods."""

    figsize: tuple[float, float]
    dpi: float
    facecolor: Color
    edgecolor: Color
    frameon: bool
    num: int | str
    clear: bool
    layout: Literal["constrained", "compressed", "tight", "none"] | None
    tight_layout: bool | dict[str, float]
    constrained_layout: bool
    nrows: int
    ncols: int
    sharex: bool | Literal["none", "all", "row", "col"]
    sharey: bool | Literal["none", "all", "row", "col"]
    squeeze: bool
    width_ratios: Sequence[float]
    height_ratios: Sequence[float]
    subplot_kw: dict[str, object]
    gridspec_kw: dict[str, object]


class DependenceOptions(TypedDict, total=False):
    """Dependence-plot options apart from feature, figure size and show."""

    bins: int | list[float] | FloatArray
    min_q: float
    max_q: float
    alpha: float


class SummaryOptions(TypedDict, total=False):
    """SHAP summary-plot options apart from plot type, class names and show."""

    max_display: int
    color: Color | Colormap
    axis_color: Color
    title: str
    alpha: float
    sort: bool
    color_bar: bool
    plot_size: tuple[float, float] | float | Literal["auto"] | None
    color_bar_label: str
    cmap: Colormap
    auto_size_plot: bool
    use_log_scale: bool


class InterpretPlotOptions(SummaryOptions, DependenceOptions):
    """Options selected according to the interpreter's plot type."""


TrainingCallback: TypeAlias = Callable[..., None]


class PoolData(Protocol):
    """The CatBoost Pool operation used by the early-stopping adapter."""

    def set_weight(self, weight: pd.Series) -> PoolData: ...


class FrameFitParams(TypedDict):
    """Required training and validation data for dataframe-based estimators."""

    X: pd.DataFrame
    y: pd.Series
    eval_set: list[tuple[pd.DataFrame, pd.Series]]


class BoostingFitParams(FrameFitParams, total=False):
    """Optional weights, evaluation metric and callbacks for early stopping."""

    sample_weight: pd.Series
    eval_sample_weight: list[pd.Series]
    eval_metric: EvalMetric | None
    callbacks: list[TrainingCallback]


class CatBoostFitParams(TypedDict):
    """CatBoost accepts Pool objects for training and validation."""

    X: PoolData
    eval_set: PoolData


class EarlyStoppingEstimator(Protocol):
    """The fit interface used after validating a supported boosting model."""

    def fit(
        self,
        *,
        X: pd.DataFrame | PoolData,
        y: pd.Series | None = None,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | PoolData,
        sample_weight: pd.Series | None = None,
        eval_sample_weight: list[pd.Series] | None = None,
        eval_metric: EvalMetric | None = None,
        callbacks: list[TrainingCallback] | None = None,
    ) -> Estimator: ...


class ResemblanceFit(Protocol):
    """Bound fit method used to forward options to a resemblance subclass."""

    def __call__(
        self,
        X1: Data,
        X2: Data,
        column_names: Sequence[Feature] | None = None,
        class_names: list[str] | None = None,
        **kwargs: Unpack[ShapOptions],
    ) -> object: ...


class SearchEstimator(Estimator, Protocol):
    """The search-CV state used after fitting a cloned search estimator."""

    estimator: Estimator
    best_params_: dict[str, object]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        groups: Labels | None = None,
    ) -> SearchEstimator: ...


class SearchFitParams(TypedDict, total=False):
    """Metadata supplied to the cloned hyperparameter search."""

    groups: Labels
