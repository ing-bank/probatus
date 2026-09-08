from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, cast, overload

import numpy as np
import pandas as pd
from shap import Explainer
from shap.explainers import TreeExplainer
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
    **shap_kwargs: Unpack[ExplainerOptions],
) -> FloatArray | tuple[FloatArray, ShapExplainer]:
    """
    Helper function to calculate the shapley values for a given model.

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
            explainer = Explainer(model, seed=random_state, **shap_kwargs)
        else:
            # Create the background data,required for non tree based models.
            # A single datapoint can passed as mask
            # (https://github.com/slundberg/shap/issues/955#issuecomment-569837201)
            if X.shape[0] < sample_size:
                sample_size = int(np.ceil(X.shape[0] * 0.2))
            else:
                pass
            mask = sample(X, sample_size, random_state=random_state)
            explainer = Explainer(model, seed=random_state, masker=mask, **shap_kwargs)

        # For tree-explainers allow for using check_additivity and approximate arguments
        if isinstance(explainer, TreeExplainer):
            shap_values = explainer.shap_values(X, check_additivity=check_additivity, approximate=approximate)

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
