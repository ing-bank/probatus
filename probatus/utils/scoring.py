from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, average_precision_score, log_loss, \
    brier_score_loss, precision_score, recall_score, jaccard_score

# We support only binary classification scorers, for now non-weighted ones
# Keys of the dict are names of metrics - values are names of scorers in sklearn SCORES dict
# Change in this parameter should be documented in docstring of volatility estimation and extending it classes
supported_scorers_dict = {'accuracy': make_scorer(accuracy_score),
                          'auc': make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True),
                          'average_precision': make_scorer(average_precision_score, needs_threshold=True),
                          'neg_log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
                          'neg_brier_score': make_scorer(brier_score_loss, greater_is_better=False,
                                                         needs_proba=True),
                          'precision': make_scorer(precision_score, average='binary'),
                          'recall': make_scorer(recall_score, average='binary'),
                          'jaccard': make_scorer(jaccard_score, average='binary')}

class Scorer:
    """
    Scores the samples model based on the provided metric name

    Args:
        metric_name: (str) name of the metric used to evaluate the model
        custom_scorer (Optional sklearn.metrics Scorer callable) object that can score samples
    """


    def __init__(self, metric_name, custom_scorer=None):
        self.metric_name = metric_name
        if custom_scorer is not None:
            self.scorer = custom_scorer
        else:
            if self.metric_name in supported_scorers_dict.keys():
                self.scorer = supported_scorers_dict[self.metric_name]
            else:
                raise(NotImplementedError('This metric name is not supported by us. Please refer to the available binary'
                                          ' classification metrics in our documentation'))

    def score(self, model, X, y):
        """
        Scores the samples model based on the provided metric name

        Args:
            model: Model to be scored, implements predict and predict_proba
            X: (array-like of shape (n_samples,n_features)) samples on which the model is scored
            y: (array-like of shape (n_samples,)) labels on which the model is scored
        """
        return self.scorer(model, X, y)