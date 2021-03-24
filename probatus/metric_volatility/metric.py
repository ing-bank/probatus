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


import pandas as pd
import numpy as np
from probatus.utils import assure_numpy_array
from probatus.metric_volatility.utils import sample_data
from sklearn.model_selection import train_test_split


def get_metric(
    X,
    y,
    clf,
    test_size,
    split_seed,
    scorers,
    train_sampling_type=None,
    test_sampling_type=None,
    train_sampling_fraction=1,
    test_sampling_fraction=1,
):
    """
    Draws random train/test sample from the data using random seed and calculates metric of interest.

    Args:
        X (np.array or pd.DataFrame):
            Dataset with features.

        y (np.array or pd.Series):
            Target of the prediction.

        clf (model object):
            Binary classification model or pipeline.

        test_size (float):
            Fraction of data used for testing the model.

        split_seed (int):
            Randomized seed used for splitting data.

        scorers (list of Scorers):
            List of Scorer objects used to score the trained model.

        train_sampling_type (str, optional):
            String indicating what type of sampling should be applied on train set:

                - `None`: indicates that no additional sampling is done after splitting data,
                - `'bootstrap'`: indicates that sampling with replacement will be performed on train data,
                - `'subsample'`: indicates that sampling without repetition will be performed  on train data.

        test_sampling_type (str, optional):
            string indicating what type of sampling should be applied on test set:

                - `None`: indicates that no additional sampling is done after splitting data
                - `'bootstrap'`: indicates that sampling with replacement will be performed on test data
                - `'subsample'`: indicates that sampling without repetition will be performed  on test data

        train_sampling_fraction (float, optional):
            Fraction of train data sampled, if sample_train_type is not None. Default value is 1.

        test_sampling_fraction (float, optional):
            Fraction of test data sampled, if sample_test_type is not None. Default value is 1.

    Returns:
        (pd.Dataframe):
            Dataframe with results for a given model trained. Rows indicate the metric measured and columns ther results
    """

    if not (isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame)):
        X = assure_numpy_array(X)
    if not (isinstance(X, np.ndarray) or isinstance(X, pd.Series)):
        y = assure_numpy_array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_seed, stratify=y)

    # Sample data based on the input arguments
    X_train, y_train = sample_data(
        X=X_train,
        y=y_train,
        sampling_type=train_sampling_type,
        sampling_fraction=train_sampling_fraction,
        dataset_name="train",
    )
    X_test, y_test = sample_data(
        X=X_test,
        y=y_test,
        sampling_type=test_sampling_type,
        sampling_fraction=test_sampling_fraction,
        dataset_name="test",
    )

    clf = clf.fit(X_train, y_train)

    results_columns = ["metric_name", "train_score", "test_score", "delta_score"]
    results = pd.DataFrame([], columns=results_columns)

    for scorer in scorers:
        score_train = scorer.score(clf, X_train, y_train)
        score_test = scorer.score(clf, X_test, y_test)
        score_delta = score_train - score_test

        results = results.append(
            pd.DataFrame(
                [[scorer.metric_name, score_train, score_test, score_delta]],
                columns=results_columns,
            )
        )
    return results
