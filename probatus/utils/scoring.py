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


from sklearn.metrics import get_scorer


def get_scorers(scoring):
    """
    Returns Scorers list based on the provided scoring.

    Args:
        scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers):
            Metrics for which the score is calculated. It can be either a name or list of names metric names and
            needs to be aligned with predefined classification scorers names in sklearn
            ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
            Another option is using probatus.utils.Scorer to define a custom metric.

    Returns:
        (list of probatus.utils.Scorer):
            List of scorers that can be used for scoring models
    """
    scorers = []
    if isinstance(scoring, list):
        for scorer in scoring:
            scorers.append(get_single_scorer(scorer))
    else:
        scorers.append(get_single_scorer(scoring))
    return scorers


def get_single_scorer(scoring):
    """
    Returns single Scorer, based on provided input in scoring argument.

    Args:
        scoring (string or probatus.utils.Scorer, optional):
            Metric for which the model performance is calculated. It can be either a metric name aligned with
            predefined classification scorers names in sklearn
            ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
            Another option is using probatus.utils.Scorer to define a custom metric.

    Returns:
        (probatus.utils.Scorer):
            Scorer that can be used for scoring models
    """
    if isinstance(scoring, str):
        return Scorer(scoring)
    elif isinstance(scoring, Scorer):
        return scoring
    else:
        raise (ValueError("The scoring should contain either strings or probatus.utils.Scorer class"))


class Scorer:
    """
    Scores a given machine learning model based on the provided metric name and optionally a custom scoring function.

    Examples:

    ```python
    from probatus.utils import Scorer
    from sklearn.metrics import make_scorer
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # Make ROC AUC scorer
    scorer1 = Scorer('roc_auc')

    # Make custom scorer with following function:
    def custom_metric(y_true, y_pred):
          return (y_true == y_pred).sum()
    scorer2 = Scorer('custom_metric', custom_scorer=make_scorer(custom_metric))

    # Prepare two samples
    feature_names = ['f1', 'f2', 'f3', 'f4']
    X, y = make_classification(n_samples=1000, n_features=4, random_state=0)
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare and fit model. Remember about class_weight="balanced" or an equivalent.
    clf = RandomForestClassifier(class_weight='balanced', n_estimators = 100, max_depth=2, random_state=0)
    clf = clf.fit(X_train, y_train)

    # Score model
    score_test_scorer1 = scorer1.score(clf, X_test, y_test)
    score_test_scorer2 = scorer2.score(clf, X_test, y_test)

    print(f'Test ROC AUC is {score_test_scorer1}, Test {scorer2.metric_name} is {score_test_scorer2}')
    ```
    """

    def __init__(self, metric_name, custom_scorer=None):
        """
        Initializes the class.

        Args:
            metric_name (str): Name of the metric used to evaluate the model.
                If the custom_scorer is not passed, the
                metric name needs to be aligned with classification scorers names in sklearn
                ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
            custom_scorer (sklearn.metrics Scorer callable, optional): Callable
                that can score samples.
        """
        self.metric_name = metric_name
        if custom_scorer is not None:
            self.scorer = custom_scorer
        else:
            self.scorer = get_scorer(self.metric_name)

    def score(self, model, X, y):
        """
        Scores the samples model based on the provided metric name.

        Args:
            model (model object):
                Model to be scored.

            X (array-like of shape (n_samples,n_features)):
                Samples on which the model is scored.

            y (array-like of shape (n_samples,)):
                Labels on which the model is scored.

        Returns:
            (float):
                Score returned by the model
        """
        return self.scorer(model, X, y)
