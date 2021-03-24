"""
Utility functions for sampling.

This module holds utility functions for sampling data. They can be used, for
example, to provide a random single typical datapoin in a technical report.
"""

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype


def sample_row(
    X: pd.DataFrame,
    filter_rows_with_na: bool = False,
    random_state: int = 42,
    max_field_len: int = 50,
) -> pd.DataFrame:
    """Sample a row from pandas dataframe.

    Extracts the column name, datatype, minimum and maximum values for each
    column in the supplied dataframe. The orientation is row-based (as opposed to `df.sample(1)`), which allows
    for better printing when a dataset contains many features. This function is usefull when providing a
    sample row in technical model documentation.

    Example:

    ```python
    from probatus.utils import sample_row
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True).get('data')
    sample = sample_row(iris, filter_rows_with_na=False, random_state=12)
    print(sample.to_markdown())
    ```

    ??? info "Example output"

        | column            | dtype   |   sample |   range_low |   range_high |
        |:------------------|:--------|---------:|------------:|-------------:|
        | sepal length (cm) | float64 |      5   |         4.3 |          7.9 |
        | sepal width (cm)  | float64 |      3.5 |         2   |          4.4 |
        | petal length (cm) | float64 |      1.3 |         1   |          6.9 |
        | petal width (cm)  | float64 |      0.3 |         0.1 |          2.5 |

    Args:
        X (DataFrame):
            Pandas DataFrame to be sampled
        filter_rows_with_na (bool):
            if true, rows with na values are not considered for sampling
        random_state (int):
            Optional random state to ensure reproducability
        max_field_len (int):
            Maximum number of characters for fields, beyond
            which any text is truncated

    Returns:
        (pd.DataFrame): A Pandas DataFrame containing the sampled row
    """
    # Input validation
    assert type(X) == pd.DataFrame, "X should be pandas DataFrame"
    assert X.empty is False, "X should not be an empty DataFrame"
    assert type(filter_rows_with_na) == bool, "filter_rows_with_na should be a boolean"
    assert type(random_state) == int, "random_state should be an integer"
    assert type(max_field_len) == int, "max_field_len should be an integer"

    # Create new empty df
    sample_df = pd.DataFrame()

    # Convert dtypes of pandas to ensure detection of data types
    X_dtypes = X.convert_dtypes()

    # Sample row from X
    sample_row = X.sample(1, random_state=random_state)
    if filter_rows_with_na:
        try:
            sample_row = X.dropna().sample(1, random_state=random_state)
        except ValueError:
            logging.info("sample_row(): No rows without NaN found, sampling from all rows..")

    # Sample every column of X
    for i, col in enumerate(sample_row.columns):
        # Extract sample from X if not all samples are nan
        sample = sample_row[col].values[0]

        # If datatype allows it, extract low and high range
        if is_numeric_dtype(X_dtypes[col]):
            low = X[col].min(skipna=True)
            high = X[col].max(skipna=True)
        else:
            low = ""
            high = ""

            # Shorten sampled datapoint if too long
            if isinstance(sample, str) and len(sample) > max_field_len:
                sample = sample[: (max_field_len // 2) - 1] + "..." + sample[(-max_field_len // 2) + 2 :]

        # Add new row to sample_df
        row_df = pd.DataFrame(
            {
                "column": [col],
                "dtype": [X[col].dtype],
                "sample": [sample],
                "range_low": [low],
                "range_high": [high],
            }
        )
        sample_df = pd.concat([sample_df, row_df], ignore_index=True)

    sample_df = sample_df.set_index(["column"])
    return sample_df
