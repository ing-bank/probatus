import warnings
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd


def assure_pandas_df(
    x: Union[List, np.ndarray, pd.DataFrame, pd.Series], column_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Converts various data types to a pandas DataFrame with specified column names.

    This function handles different input types and ensures the output is always a pandas DataFrame.
    If the input is already a DataFrame, it can optionally rename the columns.

    Args:
        x: Input data to convert to DataFrame. Can be:
           - list or list of lists
           - numpy array
           - pandas DataFrame
           - pandas Series
        column_names: Optional list of column names to use for the DataFrame.
                     If provided for an existing DataFrame, it will replace the current column names.

    Returns:
        pandas DataFrame with the data from x and specified column names (if provided)

    Raises:
        TypeError: If x is not one of the supported types
    """
    if isinstance(x, pd.DataFrame):
        if column_names is not None:
            x.columns = column_names
    elif isinstance(x, (np.ndarray, pd.Series, list)):
        x = pd.DataFrame(x, columns=column_names)

    return x


def assure_pandas_series(
    x: Union[List, np.ndarray, pd.Series], index: Optional[Union[List, np.ndarray, pd.Index]] = None
) -> pd.Series:
    """
    Converts various data types to a pandas Series with specified index.

    This function handles different input types and ensures the output is always a pandas Series.
    If the input is already a Series, it handles index alignment based on the provided index.

    Args:
        x: Input data to convert to Series. Can be:
           - list
           - numpy array
           - pandas Series
        index: Optional index to use for the Series. If x is already a Series:
               - If index matches exactly: returns x unchanged
               - If index has same values but different order: reorders x
               - If index has different values: overwrites x's index

    Returns:
        pandas Series with the data from x and specified index (if provided)

    Raises:
        TypeError: If x is not one of the supported types
    """
    if isinstance(x, pd.Series):
        if index is None:
            return x

        if isinstance(index, (list, np.ndarray)):
            index = pd.Index(index)

        current_x_index = pd.Index(x.index.values)

        # Three cases for index handling:
        if current_x_index.equals(index):
            # Case 1: If exact match then keep it as it is
            return x
        elif current_x_index.sort_values().equals(index.sort_values()):
            # Case 2: If both have the same values but in different order, then reorder
            return x[index]
        else:
            # Case 3: If indexes have different values, overwrite
            x.index = index
            return x
    elif isinstance(x, (np.ndarray, list)):
        return pd.Series(x, index=index)


def preprocess_data(
    X: Union[pd.DataFrame, List, np.ndarray],
    X_name: Optional[str] = None,
    column_names: Optional[List[str]] = None,
    verbose: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess data for machine learning models.

    This function performs several preprocessing steps:
    1. Converts input to a pandas DataFrame
    2. Warns about features with missing values (if verbose)
    3. Warns about categorical features (if verbose)
    4. Converts object dtype features to category dtype for better compatibility with models like LightGBM

    Args:
        X: Input dataset to preprocess. Can be:
           - pandas DataFrame
           - list of lists
           - numpy array

        X_name: Name of the X variable, used in warning messages.
                Defaults to "X" if not provided.

        column_names: List of feature names to use for the dataset.
                     If provided, overwrites existing feature names.
                     If not provided, uses existing names or generates default ones.

        verbose: Controls verbosity of the output:
                - 0: No warnings or prints
                - 1: Only important warnings
                - 2: All prints and warnings

    Returns:
        Tuple containing:
        - Preprocessed pandas DataFrame
        - List of column names in the DataFrame
    """
    # Set default name for warning messages
    X_name = "X" if X_name is None else X_name

    X = assure_pandas_df(X, column_names=column_names)

    if verbose > 0:
        # Check for and warn about missing values
        columns_with_missing = X.columns[X.isnull().any()].tolist()
        if columns_with_missing:
            warnings.warn(
                f"The following variables in {X_name} contains missing values {columns_with_missing}. "
                f"Make sure to impute missing or apply a model that handles them automatically."
            )

        # Check for and warn about categorical features
        categorical_features = X.select_dtypes(include=["category", "object"]).columns.tolist()
        if categorical_features:
            warnings.warn(
                f"The following variables in {X_name} contains categorical variables: "
                f"{categorical_features}. Make sure to use a model that handles them automatically or "
                f"encode them into numerical variables."
            )

    # Convert object columns to category dtype for better model compatibility
    # This is particularly helpful for models like LightGBM that can handle categorical data automatically
    object_columns = X.select_dtypes(include=["object"]).columns
    if not object_columns.empty:
        X[object_columns] = X[object_columns].astype("category")

    return X, X.columns.tolist()


def preprocess_labels(
    y: Union[pd.Series, List, np.ndarray],
    index: Optional[Union[List[int], pd.Index]] = None,
) -> pd.Series:
    """
    Prepare label data for machine learning models.

    This function converts various input types to a pandas Series with proper indexing.

    Args:
        y: Input labels to preprocess. Can be:
           - pandas Series
           - list
           - numpy array

        index: The index to use for the Series. Handling depends on input type:
               - For list/array: Sets this as the index when creating Series
               - For existing Series:
                 - If indexes match exactly: keeps as is
                 - If same values but different order: reorders
                 - If different values: overwrites current index

    Returns:
        Labels as a pandas Series with proper indexing
    """
    y = assure_pandas_series(y, index=index)

    return y
