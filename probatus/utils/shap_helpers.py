from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, cast, overload

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from shap import Explainer
from shap.explainers import LinearExplainer, TreeExplainer
from shap.maskers import Independent
from shap.utils import sample
from sklearn.pipeline import Pipeline
from typing_extensions import Unpack

from probatus._typing import (
    Array,
    Estimator,
    ExplainerOptions,
    Feature,
    FloatArray,
    ShapCalculationOptions,
    ShapExplainer,
)


@overload
def shap_calc(
    model: Estimator,
    X: pd.DataFrame | Array,
    return_explainer: Literal[False] = False,
    verbose: int = 0,
    random_state: int | None = None,
    sample_size: int = 100,
    approximate: bool = False,
    check_additivity: bool = True,
    n_jobs: int | None = 1,
    **shap_kwargs: Unpack[ExplainerOptions],
) -> FloatArray: ...


@overload
def shap_calc(
    model: Estimator,
    X: pd.DataFrame | Array,
    return_explainer: Literal[True],
    verbose: int = 0,
    random_state: int | None = None,
    sample_size: int = 100,
    approximate: bool = False,
    check_additivity: bool = True,
    n_jobs: int | None = 1,
    **shap_kwargs: Unpack[ExplainerOptions],
) -> tuple[FloatArray, ShapExplainer]: ...


@overload
def shap_calc(
    model: Estimator,
    X: pd.DataFrame | Array,
    return_explainer: bool,
    verbose: int = 0,
    random_state: int | None = None,
    sample_size: int = 100,
    approximate: bool = False,
    check_additivity: bool = True,
    n_jobs: int | None = 1,
    **shap_kwargs: Unpack[ExplainerOptions],
) -> FloatArray | tuple[FloatArray, ShapExplainer]: ...


def shap_calc(
    model: Estimator,
    X: pd.DataFrame | Array,
    return_explainer: bool = False,
    verbose: int = 0,
    random_state: int | None = None,
    sample_size: int = 100,
    approximate: bool = False,
    check_additivity: bool = True,
    n_jobs: int | None = 1,
    **shap_kwargs: Unpack[ExplainerOptions],
) -> FloatArray | tuple[FloatArray, ShapExplainer]:
    """
    Helper function to calculate the shapley values for a given model.

    By default, tree-based models are explained against a background sample of `sample_size` rows
    (SHAP's interventional approach). Passing `feature_perturbation="tree_path_dependent"` is usually
    several times faster for tree-based models, but it produces slightly different SHAP values, so
    feature rankings can change.

    Args:
        model (model):
            Trained model.

        X (pd.DataFrame or np.ndarray):
            features set.

        return_explainer (boolean):
            if True, returns a a tuple (shap_values, explainer).

        verbose (int, optional):
            Controls verbosity of the output:

            - 0 - neither prints nor warnings are shown
            - 1 - only most important warnings
            - 2 - shows all prints and all warnings.

        random_state (int, optional):
            Random state set for the nr of samples. If it is None, the results will not be reproducible. For
            reproducible results set it to an integer.

        approximate (boolean):
            if True uses shap approximations - less accurate, but very fast. It applies to tree-based explainers only.

        check_additivity (boolean):
            if False SHAP will disable the additivity check for tree-based models.

        n_jobs (int, optional):
            Number of processes used to compute SHAP values for tree-based models explained against a
            background sample. The rows are split into chunks, so the result is identical to `n_jobs=1`.
            Other explainers always run in a single process. None means 1, -1 means using all processors.

        **shap_kwargs: kwargs of the shap.Explainer

    Returns:
        (np.ndarray or tuple(np.ndarray, shap.Explainer)):
            shapley_values for the model, optionally also returns the explainer.

    """
    if isinstance(model, Pipeline):
        raise TypeError(
            "The provided model is a Pipeline. Unfortunately, the features based on SHAP do not support "
            "pipelines, because they cannot be used in combination with shap.Explainer. Please apply any "
            "data transformations before running the probatus module."
        )

    explainer: ShapExplainer
    shap_values: FloatArray | list[FloatArray]

    # Suppress warnings regarding XGboost and Lightgbm models.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if verbose <= 1 else "default")

        # For tree explainers, do not pass masker when feature_perturbation is
        # tree_path_dependent, or when X contains categorical features
        # related to issue:
        # https://github.com/slundberg/shap/issues/480
        if shap_kwargs.get("feature_perturbation") == "tree_path_dependent" or (
            isinstance(X, pd.DataFrame) and X.select_dtypes("category").shape[1] > 0
        ):
            # Calculate Shap values.
            explainer = _build_explainer(model, None, random_state, **shap_kwargs)
        else:
            # Create the background data,required for non tree based models.
            # A single datapoint can passed as mask
            # (https://github.com/slundberg/shap/issues/955#issuecomment-569837201)
            if X.shape[0] < sample_size:
                sample_size = int(np.ceil(X.shape[0] * 0.2))
            else:
                pass
            mask = sample(X, sample_size, random_state=random_state)
            explainer = _build_explainer(model, mask, random_state, **shap_kwargs)

        # For tree-explainers allow for using check_additivity and approximate arguments
        if isinstance(explainer, TreeExplainer):
            shap_values = _tree_shap_values(
                explainer, X, check_additivity=check_additivity, approximate=approximate, verbose=verbose, n_jobs=n_jobs
            )

            # From SHAP version 0.43+ https://github.com/shap/shap/pull/3121 required to
            # get the second dimension of calculated Shap values.
            if not isinstance(shap_values, list) and len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        else:
            # Calculate Shap values
            shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            warnings.warn(
                "Shap values are related to the output probabilities of class 1 for this model, instead of log odds."
            )
            shap_values = shap_values[1]

    values = cast(FloatArray, np.asarray(shap_values))
    if return_explainer:
        return values, explainer
    return values


def _build_explainer(
    model: Estimator,
    masker: pd.DataFrame | Array | None,
    random_state: int | None,
    **shap_kwargs: Unpack[ExplainerOptions],
) -> ShapExplainer:
    """
    Builds the explainer `shap.Explainer` selects automatically, without parsing tree models twice.

    The automatic selection parses a tree ensemble once to check support, then again to build the explainer.
    Requesting the tree algorithm directly skips the check. Linear models are checked first, as shap does.
    """
    linear = masker is not None and LinearExplainer.supports_model_with_masker(model, Independent(masker))
    if "algorithm" not in shap_kwargs and not linear:
        try:
            return cast(
                ShapExplainer, Explainer(model, masker=masker, seed=random_state, algorithm="tree", **shap_kwargs)
            )
        except Exception:
            # Not a tree model: fall back to the automatic selection, which also re-raises genuine errors.
            pass
    return cast(ShapExplainer, Explainer(model, masker=masker, seed=random_state, **shap_kwargs))


def _tree_shap_values(
    explainer: TreeExplainer,
    X: pd.DataFrame | Array,
    check_additivity: bool,
    approximate: bool,
    verbose: int,
    n_jobs: int | None,
) -> FloatArray:
    """
    Computes tree SHAP values, splitting the rows over processes when a background sample is used.

    SHAP's interventional computation runs in a single thread and holds the GIL, so it is parallelised
    across processes. Each row's values depend only on that row and the background sample, so the
    result is identical to a single call. Other modes are fast or already multithreaded, so run as one call.
    """
    n_chunks = min(effective_n_jobs(n_jobs), X.shape[0])
    if n_chunks <= 1 or approximate or explainer.feature_perturbation != "interventional":
        return explainer.shap_values(X, check_additivity=check_additivity, approximate=approximate)

    # With a background sample, shap sets the expected value when the explainer is built, so the
    # workers' copies of the explainer have nothing to report back.
    bounds = np.linspace(0, X.shape[0], n_chunks + 1).astype(int)
    chunks = [X.iloc[start:end] if isinstance(X, pd.DataFrame) else X[start:end] for start, end in pairwise(bounds)]
    parts = Parallel(n_jobs=n_chunks)(
        delayed(_tree_shap_values_chunk)(explainer, chunk, check_additivity, verbose) for chunk in chunks
    )
    return np.concatenate(parts, axis=0)


def _tree_shap_values_chunk(
    explainer: TreeExplainer, X: pd.DataFrame | Array, check_additivity: bool, verbose: int
) -> FloatArray:
    """Computes SHAP values for one chunk of rows in a worker process, with the caller's warning filter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if verbose <= 1 else "default")
        return explainer.shap_values(X, check_additivity=check_additivity)


def shap_to_df(
    model: Estimator,
    X: pd.DataFrame | Array,
    precalc_shap: FloatArray | None = None,
    **kwargs: Unpack[ShapCalculationOptions],
) -> pd.DataFrame:
    """
    Calculates the shap values and return the pandas DataFrame with the columns and the index of the original.

    Args:
        model (model):
            Pretrained model (Random Forest of XGBoost at the moment).

        X (pd.DataFrame or np.ndarray):
            Dataset on which the SHAP importance is calculated.

        precalc_shap (np.array):
            Precalculated SHAP values. If None, they are computed.

        **kwargs: for the function shap_calc

    Returns:
        (pd.DataFrame):
            Dataframe with SHAP feature importance per features on X dataset.
    """
    shap_values = precalc_shap if precalc_shap is not None else shap_calc(model, X, **kwargs)

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    if isinstance(X, np.ndarray) and X.ndim == 2:
        return pd.DataFrame(shap_values, columns=[f"col_{ix}" for ix in range(X.shape[1])])
    raise TypeError("X must be a dataframe or a 2d array")


def calculate_shap_importance(
    shap_values: FloatArray,
    columns: Sequence[Feature],
    output_columns_suffix: str = "",
    shap_variance_penalty_factor: float | None = None,
) -> pd.DataFrame:
    """
    Returns the average shapley value for each column of the dataframe, as well as the average absolute shap value.

    Args:
        shap_values (np.array):
            Shap values.

        columns (list of str):
            Feature names.

        output_columns_suffix (str, optional):
            Suffix to be added at the end of column names in the output.

        shap_variance_penalty_factor (int or float, optional):
            Apply aggregation penalty when computing average of shap values for a given feature.
            Results in a preference for features that have smaller standard deviation of shap
            values (more coherent shap importance). Recommend value 0.5 - 1.0.
            Formula: penalized_shap_mean = (mean_shap - (std_shap * shap_variance_penalty_factor))

    Returns:
        (pd.DataFrame):
            Mean absolute shap values and Mean shap values of features.

    """
    if shap_variance_penalty_factor is None or shap_variance_penalty_factor < 0:
        shap_variance_penalty_factor = 0
    elif not isinstance(shap_variance_penalty_factor, (float, int)):
        warnings.warn(
            "shap_variance_penalty_factor must be None, int, or float. Setting shap_variance_penalty_factor = 0"
        )
        shap_variance_penalty_factor = 0

    abs_shap_values = np.abs(shap_values)
    if np.ndim(shap_values) > 2:  # multi-class case
        abs_shap_values = np.sum(abs_shap_values, axis=0)
        shap_values = np.sum(shap_values, axis=0)

    shap_abs_mean = np.mean(abs_shap_values, axis=0)
    shap_mean = np.mean(shap_values, axis=0)
    # A disabled penalty needs no variance calculation or sample-sized variance temporary.
    penalized_shap_abs_mean = shap_abs_mean
    if shap_variance_penalty_factor != 0:
        penalized_shap_abs_mean = shap_abs_mean - (np.std(abs_shap_values, axis=0) * shap_variance_penalty_factor)

    # Sort positions so feature names (including duplicates) retain pandas' tie ordering.
    order = pd.Series(penalized_shap_abs_mean, dtype=float).sort_values(ascending=False).index

    # Construct only the public columns, directly in the output dtype.
    importance_df = pd.DataFrame(
        {
            f"mean_abs_shap_value{output_columns_suffix}": shap_abs_mean,
            f"mean_shap_value{output_columns_suffix}": shap_mean,
        },
        index=list(columns),
        dtype=float,
    )
    return importance_df.iloc[order]
