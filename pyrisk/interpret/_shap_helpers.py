import shap
import pandas as pd
import numpy as np
from ..utils import class_name_from_object, UnsupportedModelError


def shapley_calculator(model, X, approximate=False, **shap_kwargs):
    """

    Args:
        model:
        X:
        approximate:
        **shap_kwargs:

    Returns:

    """

    model_name = class_name_from_object(model)

    supported_models = ['RandomForestClassifier',"XGBClassifier"]

    if model_name not in supported_models:
        raise UnsupportedModelError("{} not supported. Please try one of the following {}".format(model_name, supported_models))

    explainer = shap.TreeExplainer(model,**shap_kwargs)

    # Calculate Shap values
    shap_values = explainer.shap_values(X, approximate=approximate)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def shap_to_dataframe(model, X, **kwargs):
    """

    Args:
        model:
        X:
        **kwargs:

    Returns:

    """


    shap_values = shapley_calculator(model, X, **kwargs)

    return pd.DataFrame(shap_values,columns=X.columns, index=X.index)



def compute_average_shap_per_column(df):
    """

    Args:
        df:

    Returns:

    """

    average_shap = pd.Series(np.mean(df.values, axis=0), index=df.columns, name='average_shap')

    average_abs_shap = pd.Series(np.mean(np.abs(df.values), axis=0), index=df.columns, name='average_abs_shap')

    return average_shap, average_abs_shap


def compute_average_shap_per_column_raw(model, X, **kwargs):
    """

    Args:
        model:
        X:
        **kwargs:

    Returns:

    """


    shap_df = shap_to_dataframe(model, X, **kwargs)

    return compute_average_shap_per_column(shap_df)
