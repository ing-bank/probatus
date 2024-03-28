import warnings

import numpy as np
import pandas as pd


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
            - 1 - only most important warnings
            - 2 - shows all prints and all warnings.


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
    if y_name is None:
        y_name = "y"

    # Make sure that y is a series with correct index
    y = assure_pandas_series(y, index=index)

    return y
