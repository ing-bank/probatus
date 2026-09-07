from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from probatus._typing import Data, Feature, Labels


def assure_pandas_df(x: Data, column_names: Sequence[Feature] | None = None) -> pd.DataFrame:
    """
    Returns x as pandas DataFrame. X can be a list, list of lists, numpy array, pandas DataFrame or pandas Series.

    Args:
        x (list, numpy array, pandas DataFrame, pandas Series): array to be tested

    Returns:
        pandas DataFrame
    """
    if isinstance(x, pd.DataFrame):
        if column_names is not None:
            x.columns = list(column_names)
    elif isinstance(x, (np.ndarray, pd.Series, list)):
        x = pd.DataFrame(x, columns=None if column_names is None else list(column_names))
    else:
        raise TypeError("Please supply a list, numpy array, pandas Series or pandas DataFrame")

    return x


def assure_pandas_series(x: Labels, index: pd.Index | list[Any] | NDArray[Any] | None = None) -> pd.Series:
    """
    Returns x as pandas Series. X can be a list, numpy array, or pandas Series.

    Args:
        x (list, numpy array, pandas DataFrame, pandas Series): array to be tested

    Returns:
        pandas Series
    """
    if isinstance(x, pd.Series):
        if index is None:
            return x
        target_index = pd.Index(index)
        current_x_index = pd.Index(x.index.values)
        if current_x_index.equals(target_index):
            # If exact match then keep it as it is
            return x
        elif current_x_index.sort_values().equals(target_index.sort_values()):
            # If both have the same values but in different order, then reorder
            return x[target_index]
        else:
            # If indexes have different values, overwrite
            x.index = target_index
            return x
    elif any([isinstance(x, (np.ndarray, list))]):
        return pd.Series(x, index=index)
    else:
        raise TypeError("Please supply a list, numpy array, pandas Series")


def preprocess_data(
    X: Data, X_name: str | None = None, column_names: Sequence[Feature] | None = None, verbose: int = 0
) -> tuple[pd.DataFrame, list[Feature]]:
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
            - 1 - only most important warnings
            - 2 - shows all prints and all warnings.


    Returns:
        (pd.DataFrame):
            Preprocessed dataset.
    """
    X_name = "X" if X_name is None else X_name

    # Make sure that X is a pd.DataFrame with correct column names
    X = assure_pandas_df(X, column_names=column_names)

    if verbose > 0:
        # Warn if missing
        columns_with_missing = X.columns[X.isnull().any()].tolist()
        if columns_with_missing:
            warnings.warn(
                f"The following variables in {X_name} contains missing values {columns_with_missing}. "
                f"Make sure to impute missing or apply a model that handles them automatically."
            )

        # Warn if categorical features and change to category
        categorical_features = X.select_dtypes(include=["category", "object"]).columns.tolist()
        # Set categorical features type to category
        if categorical_features:
            if verbose > 0:
                warnings.warn(
                    f"The following variables in {X_name} contains categorical variables: "
                    f"{categorical_features}. Make sure to use a model that handles them automatically or "
                    f"encode them into numerical variables."
                )

    # Ensure category dtype, to enable models e.g. LighGBM, handle them automatically
    object_columns = X.select_dtypes(include=["object"]).columns
    if not object_columns.empty:
        X[object_columns] = X[object_columns].astype("category")

    return X, list(X.columns)


def preprocess_labels(
    y: Labels,
    y_name: str | None = None,
    index: pd.Index | list[Any] | NDArray[Any] | None = None,
    verbose: int = 0,
) -> pd.Series:
    """
    Does basic preparation of the labels. Turns them into Series, and WARS in case the target is not binary.

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
        - 1 - only most important warnings
        - 2 - shows all prints and all warnings.

    Returns:
        (pd.Series):
            Labels in the form of pd.Series.
    """
    y_name = "y" if y_name is None else y_name

    # Make sure that y is a series with correct index
    y = assure_pandas_series(y, index=index)

    return y
