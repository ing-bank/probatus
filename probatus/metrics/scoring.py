from typing import Union, Callable, Optional
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator
import pandas as pd


def get_single_scorer(scoring: Union[str, "Scorer"]) -> "Scorer":
    """
    Returns a standardized Scorer object based on the provided input in the scoring argument.

    This function ensures that regardless of whether a string metric name or a Scorer object
    is provided, a consistent Scorer object is returned for model evaluation.

    Args:
        scoring (Union[str, 'Scorer']):
            Metric for which the model performance is calculated. It can be either:
            - A string metric name aligned with predefined classification scorers in scikit-learn
              (see: https://scikit-learn.org/stable/modules/model_evaluation.html)
            - An instance of probatus.utils.Scorer to define a custom metric

    Returns:
        Scorer:
            A Scorer object that can be used for consistent model evaluation

    Raises:
        ValueError: If the scoring parameter is neither a string nor a Scorer object
    """
    if isinstance(scoring, str):
        return Scorer(scoring)
    elif isinstance(scoring, Scorer):
        return scoring
    else:
        raise ValueError("The scoring parameter must be either a string metric name or a probatus.utils.Scorer object")


class Scorer:
    """
    A wrapper class that scores machine learning models based on a specified metric.

    This class provides a consistent interface for model evaluation, supporting both
    standard scikit-learn metrics and custom scoring functions.

    Attributes:
        metric_name (str): Name of the metric used for evaluation
        scorer (Callable): The actual scoring function used for evaluation

    Examples:

    ```python
    from probatus.metrics import Scorer
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
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=2, random_state=0)
    model = model.fit(X_train, y_train)

    # Score model
    score_test_scorer1 = scorer1.score(model, X_test, y_test)
    score_test_scorer2 = scorer2.score(model, X_test, y_test)

    print(f'Test ROC AUC is {score_test_scorer1}, Test {scorer2.metric_name} is {score_test_scorer2}')
    ```
    """

    def __init__(self, metric_name: str, custom_scorer: Optional[Callable] = None) -> None:
        """
        Initialize a Scorer object with a specified metric.

        Args:
            metric_name (str):
                Name of the metric used to evaluate the model.
                If custom_scorer is not provided, this name must match one of the
                classification scorers available in scikit-learn
                (see: https://scikit-learn.org/stable/modules/model_evaluation.html).

            custom_scorer (Optional[Callable], default=None):
                A callable scoring function that follows scikit-learn's scorer interface.
                If provided, this will be used instead of looking up a scorer by metric_name.
                Typically created using sklearn.metrics.make_scorer.
        """
        # Store the metric name for reference and reporting
        self.metric_name = metric_name

        # Determine which scorer to use - custom or from scikit-learn
        if custom_scorer is not None:
            self.scorer = custom_scorer
        else:
            # Get a standard scorer from scikit-learn based on the metric name
            try:
                self.scorer = get_scorer(self.metric_name)
            except ValueError as e:
                raise ValueError(
                    f"Metric '{self.metric_name}' not found in scikit-learn. "
                    f"Please provide a valid metric name or a custom scorer."
                ) from e

    def score(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Score a model on the provided data using the configured metric.

        This method applies the scoring function to evaluate the model's performance
        on the given feature data and target labels.

        Args:
            model (BaseEstimator):
                The trained model to be evaluated. Must implement a predict or predict_proba
                method depending on the scoring metric requirements.

            X (pd.DataFrame):
                Feature data on which to evaluate the model, of shape (n_samples, n_features).

            y (pd.Series):
                True target labels for the samples, of shape (n_samples,).

        Returns:
            float: The calculated score according to the specified metric
        """
        return self.scorer(model, X, y)
