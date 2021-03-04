import pandas as pd
import numpy as np
import logging
import pytest
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_classification 
import string 

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


def get_data(n_samples,n_numerical,n_category):
        """
        Returns a dataframe(X),target(y) with numerical and categorical features.

        Args :
            n_samples(int) : Number of samples to return.
            n_numerical(int)  : Number of numerical columns to create.
            n_category(int) : Number of categorical columns to create.

        Returns :
        X(DataFrame) : DataFrame with numerical and categorical features.
        y(Series) : Series with binary values.

        Examples:

        # Create a data with 1000 samples, 10 numerical and 5 categorical variables. 
        X,y = get_data(n_samples=1000,n_numerical=10,n_category=5)
       
        """
        #Total number of columns is the sum of numerical and categorical columns.
        no_vars = n_numerical + n_category
       
        X,y = make_classification(
            n_samples=n_samples, 
            n_features=no_vars, 
            random_state=123,class_sep=0.3)

        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile", )
        X[:,n_numerical:] = binner.fit_transform(X[:,n_numerical:])

        #Add column names.
        X = pd.DataFrame(X, columns=["f_"+str(i) for i in range(0,no_vars)])

        # Efficiently map values to another value with .map(dict)
        X.iloc[:,n_numerical:] = X.iloc[:,n_numerical:].apply(
            lambda x: x.map({i:letter for i,letter in enumerate(string.ascii_uppercase)}))
        
        return X,y
