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


import shap
import pandas as pd
import numpy as np
import warnings

def shap_calc(model, X, approximate=False, return_explainer=False, suppress_warnings=False, **shap_kwargs):
    """
    Helper function to calculate the shapley values for a given model.
    Supported models for the moment are RandomForestClassifiers and XGBClassifiers
    In case the shapley values have
    Args:
        model: pretrained model (Random Forest of XGBoost at the moment)
        X (pd.DataFrame or np.ndarray): features set
        approximate (boolean):, if True uses shap approximations - less accurate, but very fast
        return_explainer (boolean): if True, returns a a tuple (shap_values, explainer).
        suppress_warnings (boolean): If True, warnings from SHAP will be suppressed.
        **shap_kwargs: kwargs of the shap.TreeExplainer

    Returns: (np.ndarray or tuple(np.ndarray, shap.TreeExplainer)) shapley_values for the model.

    """
    # Suppress warnings regarding XGboost and Lightgbm models.
    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")

        explainer = shap.TreeExplainer(model, **shap_kwargs)
        # Calculate Shap values
        shap_values = explainer.shap_values(X, approximate=approximate)

        if isinstance(shap_values, list) and len(shap_values)==2:
            warnings.warn('Shap values are related to the output probabilities of class 1 for this model, instead of log odds.')
            shap_values = shap_values[1]

    if return_explainer:
        return shap_values, explainer
    return shap_values


def shap_to_df(model, X, precalc_shap=None, **kwargs):
    """
    Calculates the shap values and return the pandas DataFrame with the columns and the index of the original

    Args:
        model: pretrained model (Random Forest of XGBoost at the moment)
        X (pd.DataFrame or np.ndarray): features set
        precalc_shap (np.array): Precalculated SHAP values. If None, they are computed.
        **kwargs: for the function shap_calc

    Returns:

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


def calculate_shap_importance(shap_values, columns):
    """
    Returns the average shapley value for each column of the dataframe, as well as the average absolute shap value.

    Args:
        shap_values (np.array): Shap values.
        columns (list of str): Feature names.

    Returns:
        (pd.DataFrame): Mean absolute shap values and Mean shap values of features.

    """

    # Find average shap importance for neg and pos class
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    shap_mean = np.mean(shap_values, axis=0)

    # Prepare importance values in a handy df
    importance_df = pd.DataFrame({
        'mean_abs_shap_value':shap_abs_mean.tolist(),
        'mean_shap_value': shap_mean.tolist()},
        index=columns)

    # Set the correct column types
    importance_df['mean_abs_shap_value'] = importance_df['mean_abs_shap_value'].astype(float)
    importance_df['mean_shap_value'] = importance_df['mean_shap_value'].astype(float)

    importance_df = importance_df.sort_values('mean_abs_shap_value', ascending=False)

    return importance_df
