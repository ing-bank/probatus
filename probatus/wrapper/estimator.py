"""
Model wrapper classes for Probatus.

This module provides wrapper classes for machine learning models used throughout Probatus.
These classes handle model validation, preprocessing, and provide a consistent interface
for various analysis methods to interact with different types of models.
"""

import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from typing import Union, Optional

from probatus._common.data_processing import (
    get_estimator,
    is_multi_classifier,
    get_preprocessor,
)
from probatus.features._validation._parameters import _validate_model_compatibility_with_early_stopping_parameter
from probatus.wrapper.scoring import Scorer, get_single_scorer
from probatus.wrapper.base import BaseFitClass


class BaseModel(BaseFitClass):
    """
    Base class for model wrappers in Probatus.

    This class provides fundamental model validation and identification capabilities,
    serving as a foundation for specialized model wrapper classes. It handles different
    model types including scikit-learn estimators, search CV objects, and pipelines.

    Attributes:
        has_pipeline (bool): Whether the model is contained in a Pipeline.
        model (BaseEstimator): The extracted estimator from the model.
        is_search_model (bool): Whether the model is a search CV object (e.g., GridSearchCV).
        is_classifier (bool): Whether the model is a classifier.
        is_regressor (bool): Whether the model is a regressor.
        is_multi_classifier (Optional[bool]): Whether the model is a multi-class classifier.
            Initially None until determine_multi_classifier() is called.
    """

    def __init__(
        self,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
    ) -> None:
        """
        Initialize the BaseModel wrapper.

        Args:
            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to wrap. Can be a scikit-learn estimator, search CV object, or pipeline.
        """
        self.has_pipeline: bool = isinstance(model, Pipeline)
        self.estimator = get_estimator(model)
        self.preprocessor = get_preprocessor(model)
        self.is_search_model: bool = isinstance(model, BaseSearchCV)
        self.is_classifier: bool = is_classifier(self.estimator)
        self.is_regressor: bool = is_regressor(self.estimator)
        self.is_multi_classifier: Optional[bool] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BaseModel":
        """
        Fit the estimator.
        """
        # Determine if the model is a multi-class classifier
        self.determine_multi_classifier(y)

        # Fit the estimator
        self.estimator = self.estimator.fit(X, y)
        self.set_fitted()

        return self

    def determine_multi_classifier(
        self,
        y: Optional[pd.Series] = None,
    ) -> None:
        """
        Determine whether the model is a multi-class classifier.

        This method checks if the model is a multi-class classifier by examining
        the unique values in the target variable if provided.

        Args:
            y (Optional[pd.Series], optional):
                Target variable series. If provided, the number of unique values
                is used to determine if the model is multi-class.
                Defaults to None.
        """
        self.is_multi_classifier: Optional[bool] = is_multi_classifier(self.estimator, y)


class BaseScoringModel(BaseModel):
    """
    Base class for model wrappers that require a scoring metric.

    This class extends BaseModel with functionality to handle scoring metrics,
    which are used for model evaluation in various analysis methods.

    Attributes:
        All attributes from BaseModel, plus:
        scorer (Scorer): Scorer object used to evaluate model performance.
    """

    def __init__(
        self,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
        scoring: Union[str, Scorer] = "roc_auc",
    ) -> None:
        """
        Initialize the BaseScoringModel wrapper.

        Args:
            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to wrap. Can be a scikit-learn estimator, search CV object, or pipeline.

            scoring (Union[str, Scorer], optional):
                Metric used to evaluate model performance.
                Can be a string referring to a scikit-learn metric (e.g., 'roc_auc', 'accuracy')
                or a custom Scorer object.
                Defaults to "roc_auc".
        """
        super().__init__(model)
        self.scorer: Union[str, Scorer] = get_single_scorer(scoring)


class RFEModel(BaseScoringModel):
    """
    Model wrapper for Recursive Feature Elimination (RFE).

    This class extends BaseScoringModel with functionality specific to RFE,
    including cross-validation and early stopping support for compatible models.

    Attributes:
        All attributes from BaseScoringModel, plus:
        cv (Optional[BaseCrossValidator]): Cross-validation strategy for RFE.
        early_stopping_rounds (Optional[int]): Number of rounds for early stopping.
        eval_metric (Optional[str]): Evaluation metric for early stopping.
        is_early_stopping_model (bool): Whether early stopping is enabled.
    """

    def __init__(
        self,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
        cv: Optional[BaseCrossValidator],
        scoring: Union[str, Scorer] = "roc_auc",
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
    ) -> None:
        """
        Initialize the RFEModel wrapper.

        Args:
            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to wrap. Can be a scikit-learn estimator, search CV object, or pipeline.

            cv (Optional[BaseCrossValidator]):
                Cross-validation strategy for RFE. If None, no cross-validation is performed.

            scoring (Union[str, Scorer], optional):
                Metric used to evaluate model performance.
                Can be a string referring to a scikit-learn metric (e.g., 'roc_auc', 'accuracy')
                or a custom Scorer object.
                Defaults to "roc_auc".

            early_stopping_rounds (Optional[int], optional):
                Number of consecutive rounds without improvement after which training will be stopped.
                Only applicable for compatible models (XGBoost, LGBM, CatBoost).
                Defaults to None.

            eval_metric (Optional[str], optional):
                Evaluation metric to use for early stopping.
                Required if early_stopping_rounds is provided.
                Defaults to None.

        Raises:
            ValueError: If early_stopping_rounds is provided but eval_metric is not,
                or if early_stopping_rounds is not a positive integer.
            TypeError: If the model is not compatible with early stopping.
        """
        # Handle early stopping configuration
        if early_stopping_rounds:
            if not eval_metric:
                raise ValueError(
                    "Running early stopping requires both 'early_stopping_rounds' and 'eval_metric' as"
                    " parameters to be provided and supports only 'XGBoost', 'LGBM' and 'CatBoost'."
                )

            if not isinstance(early_stopping_rounds, int) or early_stopping_rounds <= 0:
                raise ValueError(f"early_stopping_rounds must be a positive integer; got {early_stopping_rounds}.")

            if not _validate_model_compatibility_with_early_stopping_parameter(model):
                raise TypeError("Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping.")

            self.is_early_stopping_model: bool = True

        super().__init__(model, scoring)
        self.cv: Optional[BaseCrossValidator] = cv
        self.early_stopping_rounds: Optional[int] = early_stopping_rounds
        self.eval_metric: Optional[str] = eval_metric
