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


def sample_data(X, y, sampling_type, sampling_fraction, dataset_name="dataset"):
    """
    Sample data.
    """
    check_sampling_input(sampling_type, sampling_fraction, dataset_name)

    if sampling_type is None:
        return X, y

    number_of_samples = np.ceil(sampling_fraction * X.shape[0]).astype(int)
    array_index = list(range(X.shape[0]))

    if sampling_type == "bootstrap":
        rows_indexes = np.random.choice(array_index, number_of_samples, replace=True)
    else:
        if sampling_fraction == 1 or number_of_samples == X.shape[0]:
            return X, y
        else:
            rows_indexes = np.random.choice(array_index, number_of_samples, replace=True)

    # Get output correctly based on the type
    if isinstance(X, pd.DataFrame):
        output_X = X.iloc[rows_indexes]
    else:
        output_X = X[rows_indexes]
    if isinstance(y, pd.DataFrame):
        output_y = y.iloc[rows_indexes]
    else:
        output_y = y[rows_indexes]

    return output_X, output_y


def check_sampling_input(sampling_type, fraction, dataset_name):
    """
    Check.
    """
    if sampling_type is not None:
        if sampling_type == "bootstrap":
            if fraction <= 0:
                raise (ValueError(f"For bootstrapping {dataset_name} fraction needs to be above 0"))
        elif sampling_type == "subsample":
            if fraction <= 0 or fraction >= 1:
                raise (ValueError(f"For bootstrapping {dataset_name} fraction needs to be be above 0 and below 1"))
        else:
            raise (ValueError("This sampling method is not implemented"))
