from sklearn.metrics import get_scorer


def get_scorers(metrics):
    """
    Returns Scorers list based on the input metrics list

    Args:
        metrics (string, list of strings, Scorer or list of Scorers): Metrics for which the scorers should be returned.
        It can be either a single metric name or list of names metric names and needs to be aligned with predefined
        classification scorers names in sklearn, see the `sklearn documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
        In case a custom metric is used, one can create own Scorer (probatus.utils) and provide as a metric.

    Returns:
        list of Scorer objects: list of scorers that can be used for scoring models
    """
    scorers = []
    if isinstance(metrics, list):
        for metric in metrics:
            scorers = _append_single_metric_to_scorers(metric, scorers)
    else:
        scorers = _append_single_metric_to_scorers(metrics, scorers)
    return scorers


def _append_single_metric_to_scorers(metric, scorers):
    """
    Appends single metric to scorers list

    Args:
        metrics (string or Scorer object): metric that need to be added to scorers list.
        scorers (list of Scorer objects): list of scorers.

    Returns:
        list of Scorer objects: list of scorers that can be used for scoring models, extended by one element.
    """
    if isinstance(metric, str):
        scorers.append(Scorer(metric))
    elif isinstance(metric, Scorer):
        scorers.append(metric)
    else:
        raise (ValueError('The metrics should contain either strings or Scorer class'))
    return scorers


class Scorer:
    """
    Scores the samples model based on the provided metric name

    Args:
        metric_name (str):  Name of the metric used to evaluate the model. If the custom_scorer is not passed, the
        metric name needs to be aligned with predefined classification scorers names in sklearn, see the
        `sklearn documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`_

        custom_scorer (Optional sklearn.metrics Scorer callable): Object that can score samples.

    Examples:
        >>> from probatus.utils import Scorer
        >>> from sklearn.metrics import make_scorer
        >>> import numpy as np
        >>> scorer1 = Scorer('roc_auc')
        >>> def custom_metric(y_true, y_pred):
        >>>     return np.sum(y_true == y_pred)
        >>> scorer2 = Scorer('custom_metric', custom_scorer=make_scorer(custom_metric))
    """

    def __init__(self, metric_name, custom_scorer=None):
        self.metric_name = metric_name
        if custom_scorer is not None:
            self.scorer = custom_scorer
        else:
            self.scorer = get_scorer(self.metric_name)

    def score(self, model, X, y):
        """
        Scores the samples model based on the provided metric name

        Args:
            model (model object): Model to be scored.

            X (array-like of shape (n_samples,n_features)):  Samples on which the model is scored.

            y (array-like of shape (n_samples,)):  Labels on which the model is scored.

        Returns:
            float: Score returned by the model
        """
        return self.scorer(model, X, y)