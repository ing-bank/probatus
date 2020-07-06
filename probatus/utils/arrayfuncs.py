import numpy as np
import pandas as pd
import numbers

from probatus.utils import DimensionalityError
import warnings


def check_1d(x):
    """
    Checks whether or not a list, numpy array, pandas dataframe, pandas series are one-dimensional.
    Returns True when check is ok, otherwise throws a `DimensionalityError`

    Args:
        x: list, numpy array, pandas dataframe, pandas series

    Returns: True or throws `DimensionalityError`

    """
    if isinstance(x, list):
        if any([isinstance(el, list) for el in x]):
            raise DimensionalityError('The input is not 1D')
        else:
            return True
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and all([isinstance(el, numbers.Number) for el in x]):
            return True
        else:
            raise DimensionalityError('The input is not 1D')
    if isinstance(x, pd.core.frame.DataFrame):
        if len(x.columns) == 1 and pd.api.types.is_numeric_dtype(x[x.columns[0]]):
            return True
        else:
            raise DimensionalityError('The input is not 1D')
    if isinstance(x, pd.core.series.Series):
        if x.ndim == 1 and pd.api.types.is_numeric_dtype(x):
            return True
        else:
            raise DimensionalityError('The input is not 1D')


def assure_numpy_array(x, assure_1d=False):
    """
    Returns x as numpy array. X can be a list, numpy array, pandas dataframe, pandas series

    Args:
        x: list, numpy array, pandas dataframe, pandas series
        assure_1d: whether or not to assure that the input x is one-dimensional

    Returns: numpy array

    """
    if assure_1d:
        _ = check_1d(x)
    if isinstance(x, list):
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pd.core.frame.DataFrame):
        if len(x.columns) == 1:
            return x.values.flatten()
        else:
            return x.values
    if isinstance(x, pd.core.series.Series):
        return x.values
    
def assure_pandas_df(x):
    """
    Returns x as pandas DataFrame. X can be a list, list of lists, numpy array, pandas DataFrame or pandas Series
    
    Args:
        x (list, numpy array, pandas DataFrame, pandas Series): array to be tested
        
    Returns:
        pandas DataFrame
    """
    if isinstance(x, pd.core.frame.DataFrame):
        return x
    elif any([isinstance(x, np.ndarray),
            isinstance(x, pd.core.series.Series),
            isinstance(x, list)]):
        return pd.DataFrame(x)
    else:
        raise TypeError("Please supply a list, numpy array, pandas Series or pandas DataFrame")

def warn_if_missing(variable, variable_name):
    """
    Checks if for missing values: if there are notify the user

    Args:
        variable (pandas.DataFrame, pandas.Series or numpy.ndarray): data to be checked for missing values.
        variable_name (str): Name of the variable checked.
    """
    warning_text = "You have missing values in your variable {}, this might cause the model to fail. Please either " \
                   "impute the missing values or use a model that can handle them e.g. XGBoost.".format(variable_name)

    if isinstance(variable, (pd.DataFrame, pd.Series)):
        if variable.isnull().values.any():
            warnings.warn(warning_text)
    if isinstance(variable, np.ndarray):
        if np.isnan(variable).any():
            warnings.warn(warning_text)