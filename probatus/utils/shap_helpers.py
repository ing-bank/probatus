# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import warnings

import numpy as np
import pandas as pd
from shap import Explainer
from shap.explainers._tree import Tree
from shap.utils import sample
from sklearn.pipeline import Pipeline


def shap_calc(
    model,
    X,
    return_explainer=False,
    verbose=0,
    sample_size=100,
    approximate=False,
    check_additivity=True,
    **shap_kwargs,
):
    """
    Helper function to calculate the shapley values for a given model.

    Args:
        model (binary model):
            Trained model.

        X (pd.DataFrame or np.ndarray):
            features set.

        return_explainer (boolean):
            if True, returns a a tuple (shap_values, explainer).

        verbose (int, optional):
            Controls verbosity of the output:

            - 0 - nether prints nor warnings are shown
            - 1 - 50 - only most important warnings
            - 51 - 100 - shows other warnings and prints
            - above 100 - presents all prints and all warnings (including SHAP warnings).

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
        raise (
            TypeError(
                "The provided model is a Pipeline. Unfortunately, the features based on SHAP do not support "
                "pipelines, because they cannot be used in combination with shap.Explainer. Please apply any "
                "data transformations before running the probatus module."
            )
        )
    # Suppress warnings regarding XGboost and Lightgbm models.
    with warnings.catch_warnings():
        if verbose <= 100:
            warnings.simplefilter("ignore")

        # For tree explainers, do not pass masker when feature_perturbation is
        # tree_path_dependent, or when X contains categorical features
        # related to issue:
        # https://github.com/slundberg/shap/issues/480
        if shap_kwargs.get("feature_perturbation") == "tree_path_dependent" or X.select_dtypes("category").shape[1] > 0:
            # Calculate Shap values.
            explainer = Explainer(model, **shap_kwargs)
        else:
            # Create the background data,required for non tree based models.
            # A single datapoint can passed as mask
            # (https://github.com/slundberg/shap/issues/955#issuecomment-569837201)
            if X.shape[0] < sample_size:
                sample_size = int(np.ceil(X.shape[0] * 0.2))
            else:
                pass
            mask = sample(X, sample_size)
            explainer = Explainer(model, masker=mask, **shap_kwargs)

        # For tree-explainers allow for using check_additivity and approximate arguments
        if isinstance(explainer, Tree):
            # Calculate Shap values
            shap_values = explainer.shap_values(X, check_additivity=check_additivity, approximate=approximate)
        else:
            # Calculate Shap values
            shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            warnings.warn(
                "Shap values are related to the output probabilities of class 1 for this model, instead of " "log odds."
            )
            shap_values = shap_values[1]

    if return_explainer:
        return shap_values, explainer
    return shap_values


def shap_to_df(model, X, precalc_shap=None, **kwargs):
    """
    Calculates the shap values and return the pandas DataFrame with the columns and the index of the original.

    Args:
        model (binary model):
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
    if precalc_shap is not None:
        shap_values = precalc_shap
    else:
        shap_values = shap_calc(model, X, **kwargs)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    elif isinstance(X, np.ndarray) and len(X.shape) == 2:
        return pd.DataFrame(shap_values, columns=[f"col_{ix}" for ix in range(X.shape[1])])

    else:
        raise NotImplementedError("X must be a dataframe or a 2d array")


def calculate_shap_importance(shap_values, columns, output_columns_suffix="", shap_variance_penalty_factor=None):
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
    # Find average shap importance for neg and pos class
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    shap_mean = np.mean(shap_values, axis=0)

    if shap_variance_penalty_factor is None:
        _shap_variance_penalty_factor = 0
    elif (
        isinstance(shap_variance_penalty_factor, float) or isinstance(shap_variance_penalty_factor, int)
    ) and shap_variance_penalty_factor >= 0:
        _shap_variance_penalty_factor = shap_variance_penalty_factor
    else:
        warnings.warn(
            "shap_variance_penalty_factor must be None, int or float. " "Setting shap_variance_penalty_factor = 0"
        )
        _shap_variance_penalty_factor = 0

    penalized_shap_abs_mean = np.mean(np.abs(shap_values), axis=0) - (
        np.std(np.abs(shap_values), axis=0) * _shap_variance_penalty_factor
    )

    # Prepare importance values in a handy df
    importance_df = pd.DataFrame(
        {
            f"mean_abs_shap_value{output_columns_suffix}": shap_abs_mean.tolist(),
            f"mean_shap_value{output_columns_suffix}": shap_mean.tolist(),
            f"penalized_mean_abs_shap_value{output_columns_suffix}": penalized_shap_abs_mean.tolist(),
        },
        index=columns,
    )

    # Set the correct column types
    importance_df[f"mean_abs_shap_value{output_columns_suffix}"] = importance_df[
        f"mean_abs_shap_value{output_columns_suffix}"
    ].astype(float)
    importance_df[f"mean_shap_value{output_columns_suffix}"] = importance_df[
        f"mean_shap_value{output_columns_suffix}"
    ].astype(float)
    importance_df[f"penalized_mean_abs_shap_value{output_columns_suffix}"] = importance_df[
        f"penalized_mean_abs_shap_value{output_columns_suffix}"
    ].astype(float)

    importance_df = importance_df.sort_values(f"penalized_mean_abs_shap_value{output_columns_suffix}", ascending=False)

    # Drop penalized column
    importance_df = importance_df.drop(columns=[f"penalized_mean_abs_shap_value{output_columns_suffix}"])

    return importance_df
