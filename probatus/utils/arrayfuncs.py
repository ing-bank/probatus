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
            raise DimensionalityError("The input is not 1D")
        else:
            return True
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and all([isinstance(el, numbers.Number) for el in x]):
            return True
        else:
            raise DimensionalityError("The input is not 1D")
    if isinstance(x, pd.core.frame.DataFrame):
        if len(x.columns) == 1 and pd.api.types.is_numeric_dtype(x[x.columns[0]]):
            return True
        else:
            raise DimensionalityError("The input is not 1D")
    if isinstance(x, pd.core.series.Series):
        if x.ndim == 1 and pd.api.types.is_numeric_dtype(x):
            return True
        else:
            raise DimensionalityError("The input is not 1D")


def assure_numpy_array(x, assure_1d=False):
    """
    Returns x as numpy array. X can be a list, numpy array, pandas dataframe, pandas series.

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


def assure_pandas_df(x, column_names=None):
    """
    Returns x as pandas DataFrame. X can be a list, list of lists, numpy array, pandas DataFrame or pandas Series.

    Args:
        x (list, numpy array, pandas DataFrame, pandas Series): array to be tested

    Returns:
        pandas DataFrame
    """
    if isinstance(x, pd.DataFrame):
        # Check if column_names are passed correctly
        if column_names is not None:
            x.columns = column_names
        return x
    elif any(
        [
            isinstance(x, np.ndarray),
            isinstance(x, pd.core.series.Series),
            isinstance(x, list),
        ]
    ):
        return pd.DataFrame(x, columns=column_names)
    else:
        raise TypeError("Please supply a list, numpy array, pandas Series or pandas DataFrame")


def assure_pandas_series(x, index=None):
    """
    Returns x as pandas Series. X can be a list, numpy array, or pandas Series.

    Args:
        x (list, numpy array, pandas DataFrame, pandas Series): array to be tested

    Returns:
        pandas Series
    """
    if isinstance(x, pd.Series):
        if isinstance(index, list) or isinstance(index, np.ndarray):
            index = pd.Index(index)
        current_x_index = pd.Index(x.index.values)
        if current_x_index.equals(index):
            # If exact match then keep it as it is
            return x
        elif current_x_index.sort_values().equals(index.sort_values()):
            # If both have the same values but in different order, then reorder
            return x[index]
        else:
            # If indexes have different values, overwrite
            x.index = index
            return x
    elif any([isinstance(x, np.ndarray), isinstance(x, list)]):
        return pd.Series(x, index=index)
    else:
        raise TypeError("Please supply a list, numpy array, pandas Series")


def check_numeric_dtypes(x):
    """
    Checks if all entries in an array are of a data type that can be interpreted as numeric (int, float or bool).

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


def preprocess_data(X, X_name=None, column_names=None, verbose=0):
    """
    Preprocess data.

    Does basic preprocessing of the data: Transforms to DataFrame, Warns which features have missing variables,
    and transforms object dtype features to category type, such that LightGBM handles them by default.

    Args:
        X (pd.DataFrame, list of lists, np.array):
            Provided dataset.

        X_name (str, optional):
            Name of the X variable, that will be printed in the warnings.

        column_names (list of str, optional):
            List of feature names of the provided samples. If provided it will be used to overwrite the existing
            feature names. If not provided the existing feature names are used or default feature names are
            generated.

        verbose (int, optional):
            Controls verbosity of the output:

            - 0 - neither prints nor warnings are shown
            - 1 - 50 - only most important warnings regarding data properties are shown (excluding SHAP warnings)
            - 51 - 100 - shows most important warnings, prints of the feature removal process
            - above 100 - presents all prints and all warnings (including SHAP warnings).

    Returns:
        (pd.DataFrame):
            Preprocessed dataset.
    """
    if X_name is None:
        X_name = "X"

    # Make sure that X is a pd.DataFrame with correct column names
    X = assure_pandas_df(X, column_names=column_names)

    # Warn if missing
    columns_with_missing = [column for column in X.columns if X[column].isnull().values.any()]
    if len(columns_with_missing) > 0:
        if verbose > 0:
            warnings.warn(
                f"The following variables in {X_name} contains missing values {columns_with_missing}. "
                f"Make sure to impute missing or apply a model that handles them automatically."
            )

    # Warn if categorical features and change to category
    indices_categorical_features = [
        column[0] for column in enumerate(X.dtypes) if column[1].name in ["category", "object"]
    ]
    categorical_features = list(X.columns[indices_categorical_features])

    # Set categorical features type to category
    if len(categorical_features) > 0:
        if verbose > 0:
            warnings.warn(
                f"The following variables in {X_name} contains categorical variables: "
                f"{categorical_features}. Make sure to use a model that handles them automatically or "
                f"encode them into numerical variables."
            )

        # Ensure category dtype, to enable models e.g. LighGBM, handle them automatically
        for categorical_feature in categorical_features:
            if X[categorical_feature].dtype.name == "object":
                X[categorical_feature] = X[categorical_feature].astype("category")
    return X, X.columns.tolist()


def preprocess_labels(y, y_name=None, index=None, verbose=0):
    """
    Does basic preparation of the labels. Turns them into Series, and wars in case the target is not binary.

    Args:
    y (pd.Series, list, np.array):
        Provided labels.

    y_name (str, optional):
        Name of the y variable, that will be printed in the warnings.

    index (list of int or pd.Index, optional):
            The index correct index that should be used for y. In case y is a list or np.array, the index is set when
            creating pd.Series. In case it is pd.Series already, if the indexes consist of the same values, the y is
            going to be ordered based on provided index, otherwise, the current index of y is overwritten by index
            argument.

    verbose (int, optional):
        Controls verbosity of the output:

        - 0 - neither prints nor warnings are shown
        - 1 - 50 - only most important warnings regarding data properties are shown (excluding SHAP warnings)
        - 51 - 100 - shows most important warnings, prints of the feature removal process
        - above 100 - presents all prints and all warnings (including SHAP warnings).

    Returns:
        (pd.Series):
            Labels in the form of pd.Series.
    """
    if y_name is None:
        y_name = "y"

    # Make sure that y is a series with correct index
    y = assure_pandas_series(y, index=index)

    # Warn if not binary labels
    if len(y.unique()) != 2:
        if verbose > 0:
            warnings.warn(
                f"The labels in {y_name} contain {y.unique()} unique values. The features in probatus support"
                f" binary classification models, thus, the feature might not work correctly."
            )
    return y
