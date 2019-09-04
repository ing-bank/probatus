import shap
import pandas as pd
import numpy as np
from ..utils import class_name_from_object, UnsupportedModelError


def shap_calc(model, X, approximate=False, **shap_kwargs):
    """
    Helper function to calculate the shapley values for a given model.
    Supported models for the moment are RandomForestClassifiers and XGBClassifiers
    In case the shapley values have
    Args:
        model: pretrained model (Random Forest of XGBoost at the moment)
        X (pd.DataFrame or np.ndarray): features set
        approximate: boolean, if True uses shap approximations - less accurate, but very fast
        **shap_kwargs: kwargs of the shap.TreeExplainer

    Returns: (np.ndarray) shapley_values for the model

    """

    model_name = class_name_from_object(model)

    supported_models = ['RandomForestClassifier', 'XGBClassifier']

    if model_name not in supported_models:
        raise UnsupportedModelError(
            "{} not supported. Please try one of the following {}".format(model_name, supported_models))

    explainer = shap.TreeExplainer(model, **shap_kwargs)

    # Calculate Shap values
    shap_values = explainer.shap_values(X, approximate=approximate)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def shap_to_df(model, X, **kwargs):
    """
    Calculates the shap values and return the pandas DataFrame with the columns and the index of the original

    Args:
        model: pretrained model (Random Forest of XGBoost at the moment)
        X (pd.DataFrame or np.ndarray): features set
        **kwargs: for the function shap_calc

    Returns:

    """

    shap_values = shap_calc(model, X, **kwargs)
    if isinstance(X,pd.DataFrame):
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    elif isinstance(X,np.ndarray) and len(X.shape)==2:
        return pd.DataFrame(shap_values, columns=["col_{}" for ix in range(X.shape[1])])

    else: raise NotImplementedError("X must be a dataframe or a 2d array")


def mean_shap(df):
    """
    Returns the average shapley value for each column of the dataframe, as well as the average absolute shap value

    Args:
        df: df of shapley values (output of function shap_to_df)

    Returns: tuple of pd.Series: average shap per column, average absolute value shap per column

    """

    average_shap = pd.Series(np.mean(df.values, axis=0), index=df.columns, name='average_shap')

    average_abs_shap = pd.Series(np.mean(np.abs(df.values), axis=0), index=df.columns, name='average_abs_shap')

    return average_shap, average_abs_shap


def mean_shap_raw(model, X, **kwargs):
    """
    Returns the average shapley value for each column of the dataframe, as well as the average absolute shap value.

    Args:
        model: pretrained model (Random Forest of XGBoost at the moment)
        X (pd.DataFrame or np.ndarray): features set
        **kwargs:
            approximate  (boolean) if True uses shap approximations - less accurate, but very fast
            **kwargs of the shap.TreeExplainer

    Returns: tuple of pd.Series: average shap per column, average absolute value shap per column

    """

    shap_df = shap_to_df(model, X, **kwargs)

    return mean_shap(shap_df)
