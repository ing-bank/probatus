import warnings

import numpy as np
import pandas as pd
from shap import Explainer
from shap.explainers import TreeExplainer
from shap.utils import sample
from sklearn.pipeline import Pipeline


def shap_calc(
    model,
    X,
    return_explainer=False,
    verbose=0,
    random_state=None,
    sample_size=100,
    approximate=False,
    check_additivity=True,
    **shap_kwargs,
):
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

    # Suppress warnings regarding XGboost and Lightgbm models.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if verbose <= 1 else "default")

        # For tree explainers, do not pass masker when feature_perturbation is
        # tree_path_dependent, or when X contains categorical features
        # related to issue:
        # https://github.com/slundberg/shap/issues/480
        if shap_kwargs.get("feature_perturbation") == "tree_path_dependent" or X.select_dtypes("category").shape[1] > 0:
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

    try:
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    except AttributeError:
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            return pd.DataFrame(shap_values, columns=[f"col_{ix}" for ix in range(X.shape[1])])
        else:
            raise TypeError("X must be a dataframe or a 2d array")


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
    if shap_variance_penalty_factor is None or shap_variance_penalty_factor < 0:
        shap_variance_penalty_factor = 0
    elif not isinstance(shap_variance_penalty_factor, (float, int)):
        warnings.warn(
            "shap_variance_penalty_factor must be None, int, or float. Setting shap_variance_penalty_factor = 0"
        )
        shap_variance_penalty_factor = 0

    abs_shap_values = np.abs(shap_values)
    if np.ndim(shap_values) > 2:  # multi-class case
        sum_abs_shap = np.sum(abs_shap_values, axis=0)
        sum_shap = np.sum(shap_values, axis=0)
        shap_abs_mean = np.mean(sum_abs_shap, axis=0)
        shap_mean = np.mean(sum_shap, axis=0)
        penalized_shap_abs_mean = shap_abs_mean - (np.std(sum_abs_shap, axis=0) * shap_variance_penalty_factor)
    else:
        # Find average shap importance for neg and pos class
        shap_abs_mean = np.mean(abs_shap_values, axis=0)
        shap_mean = np.mean(shap_values, axis=0)
        penalized_shap_abs_mean = shap_abs_mean - (np.std(abs_shap_values, axis=0) * shap_variance_penalty_factor)

    # Prepare the values in a df and set the correct column types
    importance_df = pd.DataFrame(
        {
            f"mean_abs_shap_value{output_columns_suffix}": shap_abs_mean,
            f"mean_shap_value{output_columns_suffix}": shap_mean,
            f"penalized_mean_abs_shap_value{output_columns_suffix}": penalized_shap_abs_mean,
        },
        index=columns,
    ).astype(float)

    importance_df = importance_df.sort_values(f"penalized_mean_abs_shap_value{output_columns_suffix}", ascending=False)

    # Drop penalized column
    importance_df = importance_df.drop(columns=[f"penalized_mean_abs_shap_value{output_columns_suffix}"])

    return importance_df
