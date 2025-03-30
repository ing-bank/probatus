from probatus.core import BaseFitComputePlotClass
from probatus.utils import (
    Scorer,
    get_pipeline_estimator_and_preprocessor,
    get_single_scorer,
    preprocess_data,
    preprocess_labels,
)


import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


import warnings
from typing import Any, List, Literal, Tuple, cast, Optional, Union


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
        model: Union[BaseEstimator, Pipeline],
        scoring: Union[str, Scorer] = "roc_auc",
        test_prc: float = 0.25,
        n_jobs: int = 1,
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
                or a `probatus.utils.Scorer` object for custom scoring.
                Defaults to `"roc_auc"`.

            test_prc (float, optional):
                Fraction of data used for testing, in the range (0, 1].
                Defaults to `0.25`.

            n_jobs (int, optional):
                Number of parallel processes to use. Set to `-1` to use all available cores.
                Defaults to `1`.

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
        self.model, self.preprocessor = get_pipeline_estimator_and_preprocessor(model)
        self.test_prc = test_prc
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.scorer = get_single_scorer(scoring)
        self.fitted = False
        self.report_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
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
        # Set class names for the two samples
        self.class_names = ["First Sample", "Second Sample"] if class_names is None else class_names

        # Transform data if model is a Pipeline
        if self.preprocessor is not None:
            column_names = X1.columns if column_names is None else column_names
            X1 = self.preprocessor.transform(X1)
            X2 = self.preprocessor.transform(X2)

        self.X1, self.column_names = preprocess_data(X1, X_name="X1", column_names=column_names, verbose=self.verbose)
        self.X2, _ = preprocess_data(X2, X_name="X2", column_names=column_names, verbose=self.verbose)

        # Create binary classification dataset:
        # - Combine both samples
        # - Label X1 as class 0, X2 as class 1
        self.X = pd.DataFrame(pd.concat([self.X1, self.X2], axis=0), columns=self.column_names).reset_index(drop=True)
        self.y = pd.Series(
            np.concatenate(
                [
                    np.zeros(self.X1.shape[0]),  # Label 0 for all rows from X1
                    np.ones(self.X2.shape[0]),  # Label 1 for all rows from X2
                ]
            )
        ).reset_index(drop=True)

        # Ensure labels are properly formatted
        self.y = preprocess_labels(self.y, index=self.X.index)

        # Split data into training and test sets, stratifying by class to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_prc,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.y,
        )

        # Train the model to distinguish between the two samples
        self.model.fit(self.X_train, self.y_train)

        self.train_score = np.round(self.scorer.score(self.model, self.X_train, self.y_train), 3)
        self.test_score = np.round(self.scorer.score(self.model, self.X_test, self.y_test), 3)

        self.results_text = (
            f"Train {self.scorer.metric_name}: {self.train_score},\nTest {self.scorer.metric_name}: {self.test_score}."
        )
        if self.verbose > 0:
            logger.info(f"Finished model training: \n{self.results_text}")

        # Warn about potential overfitting
        if self.verbose > 0 and self.train_score > self.test_score:
            warnings.warn(
                f"Train {self.scorer.metric_name} > Test {self.scorer.metric_name}, which might indicate "
                f"overfitting. This could lead to misleading feature importance. "
                f"Consider adding regularization to the model."
            )

        self.fitted = True
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
        self._check_if_fitted()

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
