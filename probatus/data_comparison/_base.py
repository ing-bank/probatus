from probatus.wrapper import Scorer, BaseFitComputePlotClass

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

import warnings
from typing import Any, List, Literal, Tuple, Optional, Union, cast

from probatus.wrapper.data import ImportanceDataManager
from probatus.wrapper.estimator import BaseScoringModel


class BaseResemblanceModel(BaseFitComputePlotClass):
    """
    Base class for models that check the similarity between two samples.

    This class provides the foundation for analyzing whether two samples differ from each other,
    which is useful for detecting non-stationarity between training and test data.

    This is an abstract base class that needs to be extended with:
    1. A fit() method that implements how the data is split, trained, and evaluated
    2. A method to calculate feature importance

    Attributes:
        model (BaseEstimator): ML model used to distinguish between samples
        test_prc (float): Percentage of data used for testing
        n_jobs (int): Number of parallel jobs to run
        random_state (Optional[int]): Random seed for reproducibility
        verbose (int): Controls output verbosity
        scorer: Scoring metric for model evaluation
        fitted (bool): Boolean indicating if the model has been fitted
        X1 (pd.DataFrame): First sample data (set after fitting)
        X2 (pd.DataFrame): Second sample data (set after fitting)
        X (pd.DataFrame): Combined dataset (set after fitting)
        y (pd.Series): Binary labels for combined dataset (set after fitting)
        column_names (List[str]): Feature names (set after fitting)
        class_names (List[str]): Names for the two classes (set after fitting)
    """

    def __init__(
        self,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
        scoring: Union[str, Scorer] = "roc_auc",
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initializes the BaseResemblanceModel.

        Args:
            model (Union[BaseEstimator, Pipeline]):
                A regression or classification model (or pipeline) that must implement
                `fit()` and either `predict()` or `predict_proba()`.

            scoring (str, optional):
                Metric used to evaluate model performance.
                Can be a string referring to a scikit-learn classification metric
                (see: https://scikit-learn.org/stable/modules/model_evaluation.html)
                or a `probatus.wrapper.Scorer` object for custom scoring.
                Defaults to `"roc_auc"`.

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Important warnings only.
                - `2`: All warnings and detailed logs.
                Defaults to `0`.

            random_state (Optional[int], optional):
                Random seed for reproducibility. Use an integer for deterministic results
                or `None` for non-reproducible behavior. Defaults to `None`.
        """
        self.model: BaseScoringModel = BaseScoringModel(model, scoring=scoring)
        self.random_state: Optional[int] = random_state
        self.verbose: Literal[0, 1, 2] = verbose
        self.is_fitted: bool = False
        self.report_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        X_test_size: float = 0.25,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> "BaseResemblanceModel":
        """
        Prepare data and fit the model to distinguish between two samples.

        This method performs the following steps:
        1. Assigns class labels to each sample (0 for X1, 1 for X2)
        2. Combines and preprocesses the data
        3. Splits the data into training and test sets
        4. Trains the model and evaluates performance

        Args:
            X1 (pd.DataFrame): First sample to compare.
                Must have same number of columns as X2.
                Shape: (n_samples_1, n_features)

            X2 (pd.DataFrame): Second sample to compare.
                Must have same number of columns as X1.
                Shape: (n_samples_2, n_features)

            X_test_size (float, optional):
                Fraction of data used for testing, in the range (0, 1].
                Defaults to `0.25`.

            column_names (Optional[List[str]], optional): Feature names for the samples.
                If provided, overwrites existing feature names.
                If not provided, uses existing names or generates default ones.
                Length must match number of features.

            class_names (Optional[List[str]], optional): Names for the two classes/samples.
                Default is ["First Sample", "Second Sample"].
                Must be a list of length 2.

        Returns:
            BaseResemblanceModel: The fitted model instance.

        Raises:
            ValueError: If input data dimensions don't match.
            ValueError: If column_names length doesn't match number of features.
            ValueError: If class_names is provided but not of length 2.
            Warning: If train score is significantly higher than test score,
                    indicating potential overfitting.
        """
        # Prepare data for processing
        self.data_manager: ImportanceDataManager = ImportanceDataManager(
            X1=X1, X2=X2, model=self.model, X_test_size=X_test_size, column_names=column_names, class_names=class_names
        )

        # Fit the estimator
        self.model.fit(self.data_manager.X_train, self.data_manager.y_train)
        self.set_fitted()

        # Calculate scores
        self.train_score: float = self.model.scorer.score(self.data_manager.X_train, self.data_manager.y_train)
        self.test_score: float = self.model.scorer.score(self.data_manager.X_test, self.data_manager.y_test)

        if self.verbose > 0:
            results_text = (
                f"Train {self.model.scorer.scoring.metric_name}: {round(self.train_score, 4)},"
                + f"\nTest {self.model.scorer.scoring.metric_name}: {round(self.test_score, 4)}."
            )

            logger.info(f"Finished model training: \n{results_text}")

            # Warn about potential overfitting
            if self.train_score > self.test_score:
                warnings.warn(
                    f"Train {self.model.scorer.scoring.metric_name} > Test {self.model.scorer.scoring.metric_name}, which might indicate "
                    f"overfitting. This could lead to misleading feature importance. "
                    f"Consider adding regularization to the model."
                )

        return self

    def compute(self, return_scores: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
        """
        Return the feature importance report and optionally the model scores.

        This method returns the feature importance analysis results after the model has been fitted.
        The report format depends on the specific implementation (e.g., SHAP or Permutation importance).

        Args:
            return_scores (bool, optional): Whether to return model performance scores.
                If True, returns a tuple with (feature_importances, train_score, test_score).
                If False (default), returns only feature_importances.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
                If return_scores=False:
                    pd.DataFrame: Feature importance report with feature-wise metrics
                If return_scores=True:
                    Tuple containing:
                    - pd.DataFrame: Feature importance report
                    - float: Training score using the specified metric
                    - float: Test score using the specified metric

        Raises:
            ValueError: If the model has not been fitted yet. Call fit() before computing results.
        """
        self.check_if_fitted()

        if return_scores:
            return self.report_df, cast(float, self.train_score), cast(float, self.test_score)
        else:
            return self.report_df

    def fit_compute(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        return_scores: bool = False,
        **fit_kwargs: Any,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
        """
        Fit the model and compute feature importance in a single step.

        This is a convenience method that combines the functionality of fit() and compute().
        It first fits the model to distinguish between the two samples, then computes
        feature importance metrics.

        Args:
            X1 (pd.DataFrame): First sample to compare.
                Must have same number of columns as X2.
                Shape: (n_samples_1, n_features)

            X2 (pd.DataFrame): Second sample to compare.
                Must have same number of columns as X1.
                Shape: (n_samples_2, n_features)

            column_names (Optional[List[str]], optional): Feature names for the samples.
                If provided, overwrites existing feature names.
                If not provided, uses existing names or generates default ones.
                Length must match number of features.

            class_names (Optional[List[str]], optional): Names for the two classes/samples.
                Default is ["First Sample", "Second Sample"].
                Must be a list of length 2.

            return_scores (bool, optional): Whether to return model performance scores.
                If True, returns tuple (feature_importances, train_score, test_score).
                If False (default), returns only feature_importances.

            **fit_kwargs (Any): Additional keyword arguments passed to the fit() method.
                These vary based on the specific implementation (e.g., SHAP parameters).

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
                If return_scores=False:
                    pd.DataFrame: Feature importance report with feature-wise metrics
                If return_scores=True:
                    Tuple containing:
                    - pd.DataFrame: Feature importance report
                    - float: Training score using the specified metric
                    - float: Test score using the specified metric

        Raises:
            ValueError: If input data dimensions don't match or other validation errors.
        """
        self.fit(X1, X2, column_names=column_names, class_names=class_names, **fit_kwargs)
        return self.compute(return_scores=return_scores)

    def plot(self, **kwargs: Any) -> Any:
        """
        Abstract method for plotting results.

        This method should be implemented by subclasses to visualize the feature importance results.
        Each implementation should create an appropriate visualization based on its analysis method
        (e.g., SHAP plots or permutation importance plots).

        Args:
            **kwargs (Any): Additional keyword arguments passed to the specific plotting implementation.
                These vary based on the specific implementation.

        Returns:
            Any: The plot object or visualization result.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
                Each subclass should implement its own visualization logic.
        """
        raise NotImplementedError("Plot method has not been implemented.")
