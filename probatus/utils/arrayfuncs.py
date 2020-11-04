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
        if len(x.columns) == 1 and pd.api.types.is_numeric_dtype(
                x[x.columns[0]]):
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
    elif any([
            isinstance(x, np.ndarray),
            isinstance(x, pd.core.series.Series),
            isinstance(x, list)
    ]):
        return pd.DataFrame(x)
    else:
        raise TypeError(
            "Please supply a list, numpy array, pandas Series or pandas DataFrame"
        )

def assure_column_names_consistency(column_names, df):
    """
    Ensure that the column names are correct. If they are None, then, the column names from df are taken. Otherwise,
    the function checks if the passed list has the correct shape, to be consistent with the size of the df.

    Args:
        column_names (None, or list of str): List of column names to use for the df.
        df (pd.DataFrame): Dataset.

    Returns:
        (list of str): List of column names.
    """
    # Check if column_names are passed correctly
    if column_names is None:
        # Checking if original X1 was a df then taking its column names
        if isinstance(df, pd.DataFrame):
            column_names = df.columns.tolist()
        # Otherwise make own feature names
        else:
            column_names = ['column_{}'.format(idx) for idx in range(df.shape[1])]
    else:
        if isinstance(column_names, list):
            if len(column_names) == df.shape[1]:
                column_names = column_names
            else:
                raise (ValueError("Passed column_names have different dimensionality than input samples. "
                                  "The dimensionality of column_names is {} and first sample {}".
                                  format(len(column_names), df.shape[1])))
        else:
            raise (TypeError("Passed column_names must be a list"))
    return column_names


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


def check_numeric_dtypes(x):
    """
    Checks if all entries in an array are of a data type that can be interpreted as numeric (int, float or bool)
    
    Args:
        x (np.ndarray or pd.Series, list): array to be checked
    
    Returns:
        x: unchanged input array
        
    Raises:
        TypeError: if not all elements are of numeric dtypes
    """
    x = assure_numpy_array(x)
    allowed_types = [bool, int, float]
    
    for element in np.nditer(x):
        if type(element.item()) not in allowed_types:
            raise TypeError("Please supply an array with only floats, ints or booleans")
    return x


