import pandas as pd
import numpy as np


def generate_MCAR(df,missing):
    """
    Generate missing values completely at random for dataframe df

    Args:
        df: input dataframe where some values will be masked
        missings: (float or dict)
            - float ( must be a fraction between 0 and 1 - both inclusive), then it will apply this fraction of missing
            values on the whole dataset.
            - dict:
                - keys: column names to mask values
                - values: fraction of missing values for this column

    Returns:
        pd.DataFrame: same as the input dataframe, but with some values masked based on the missing variable

    Examples:

        # Apply 20% missing values over all the columns
        miss_rand = generate_MCAR(data, missing=0.2)

        # Use the dictionary
        missing_vals = {"PAY_0":0.3,"PAY_5": 0.5}
        miss_rand = generate_MCAR(data, missing=missing_vals)

    """

    df = df.copy()

    if type(missing)==float and missing<=1 and missing>=0:
        df = df.mask(np.random.random(df.shape) < missing)
    elif type(missing)==dict:
        for k,v in missing.items():
            df[k] = df[k].mask(np.random.random(df.shape[0]) < v)

    else:
        raise ValueError("missing must be float within range [0.1] or dict")

    return df


def generate_MNAR(df,missing, conditions, missing_false = None):
    """
     Generate missing values not at random for dataframe df

     Missing values following the strategy are generated at random when the condition is satisfied.

     A different method can be added for cases when the condition is false, using the missing_false variable.
     The default value None does not do any type of inputattion when the condition is false

     Args:
         df: input dataframe where some values will be masked
         missings: (float or dict)
             - float ( must be a fraction between 0 and 1 - both inclusive), then it will apply this fraction of missing
             values on the whole dataset.
             - dict:
                 - keys: column names to mask values
                 - values: fraction of missing values for this column
         conditions: pd.Series (boolean): series with same index ad df with boolean values
         missing_false: default None. Add missing values in case the condition False.
            If None, then no masking is performed.
            - float ( must be a fraction between 0 and 1 - both inclusive), then it will apply this fraction of missing
             values on the whole dataset.
             - dict:
                 - keys: column names to mask values
                 - values: fraction of missing values for this column

     Returns:
         pd.DataFrame: same as the input dataframe, but with some values masked based on the missing variable

    

     """

    df_true = df[conditions]
    df_false= df[~conditions]

    df_true = generate_MCAR(df_true,missing)

    if missing_false is not None:
        df_false = generate_MCAR(df_false, missing_false)

    out = pd.concat([df_true,df_false])

    return out
