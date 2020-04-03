from sklearn.metrics import make_scorer, get_scorer


class Scorer:
    """
    Scores the samples model based on the provided metric name

    Args:
        metric_name (str):  Name of the metric used to evaluate the model. If the custom_scorer is not passed, the
        metric name needs to be aligned with predefined classification scorers names in sklearn, see the
        `sklearn documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`_

        custom_scorer (Optional sklearn.metrics Scorer callable): Object that can score samples.
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