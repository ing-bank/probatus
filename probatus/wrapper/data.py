"""
Data management classes for Probatus.

This module provides data management classes for different Probatus use cases.
These classes handle data preprocessing, validation, and transformation to ensure
consistent interfaces for various analysis methods.
"""

from abc import abstractmethod
from typing import Iterable, List, Optional, Literal, Tuple, Union, Any
import warnings
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.calibration import check_cv
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection._search import BaseSearchCV

from probatus._common.array_operations import assure_pandas_series, preprocess_data, preprocess_labels
from probatus._common.data_processing import get_preprocessor, preprocess_class_names, get_estimator
from probatus.wrapper.estimator import BaseModel


class BaseDataManager:
    """
    Base class for data management in Probatus.

    This class provides common functionality for data preprocessing, validation,
    and transformation. It serves as a foundation for specialized data manager classes
    that handle different model analysis scenarios.

    Attributes:
        is_regressor (bool): Whether the model is a regression model.
        preprocessor (Optional[Pipeline]): Preprocessing pipeline extracted from the model if available.
        verbose (Literal[0, 1, 2]): Controls the verbosity level for warnings and information.
    """

    def __init__(
        self,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
        verbose: Literal[0, 1, 2] = 0,
    ) -> None:
        """
        Initialize the BaseDataManager.

        Args:
            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to analyze. Can be a scikit-learn estimator, search CV object, or pipeline.

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Important warnings only.
                - `2`: All warnings and detailed logs.
                Defaults to `0`.
        """
        self.is_regressor: bool = is_regressor(get_estimator(model))
        self.preprocessor: Optional[Pipeline] = get_preprocessor(model)
        self.verbose: Literal[0, 1, 2] = verbose

    @staticmethod
    def _preprocess_features_and_column_names(
        X: pd.DataFrame,
        X_name: Literal["X", "X_train", "X_test"],
        column_names: Optional[List[str]] = None,
        verbose: Literal[0, 1, 2] = 0,
        preprocessor: Optional[Pipeline] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess features and ensure proper column names.

        This method handles the transformation of input data using a preprocessor if provided,
        and ensures that the resulting DataFrame has appropriate column names.

        Args:
            X (pd.DataFrame):
                Input features to preprocess.

            X_name (Literal["X", "X_train", "X_test"], optional):
                Name of the input features.
                Defaults to "X".

            column_names (Optional[List[str]], optional):
                Custom column names to use. If None, uses existing column names or generates default ones.
                Defaults to None.

            verbose (Literal[0, 1, 2], optional):
                Controls verbosity of warning messages.
                Defaults to 0.

            preprocessor (Optional[Pipeline], optional):
                Scikit-learn pipeline to use for preprocessing.
                If provided, X will be transformed using this pipeline before further processing.
                Defaults to None.

        Returns:
            Tuple[pd.DataFrame, List[str]]:
                A tuple containing:
                - Preprocessed DataFrame
                - List of column names
        """
        if preprocessor is None:
            return preprocess_data(X, X_name=X_name if X_name else "X", column_names=column_names, verbose=verbose)
        else:
            X: pd.DataFrame = preprocessor.transform(X)
            return preprocess_data(X, X_name=X_name if X_name else "X", column_names=column_names, verbose=verbose)

    @staticmethod
    def _preprocess_class_names(
        y: pd.Series,
        class_names: Optional[List[str]] = None,
        is_regressor: bool = False,
    ) -> List[str]:
        """
        Preprocess class names for model interpretation.

        This method handles the creation of appropriate class names based on the target variable
        and whether the model is a regression or classification model.

        Args:
            y (pd.Series):
                Target variable series.

            class_names (Optional[List[str]], optional):
                Custom class names to use. If provided, these names are used directly.
                If None, appropriate names are generated based on the target variable.
                Defaults to None.

            is_regressor (bool, optional):
                Whether the model is a regression model.
                Defaults to False.

        Returns:
            List[str]: List of class names for interpretation.
        """
        if class_names is None:
            return preprocess_class_names(y, class_names, is_regressor)
        else:
            return class_names

    @staticmethod
    def _preprocess_labels(
        y: pd.Series,
        class_names: Optional[List[str]] = None,
        is_regressor: bool = False,
    ) -> pd.Series:
        """
        Preprocess target labels for model training and evaluation.

        This method ensures target labels are properly formatted and indexed.

        Args:
            y (pd.Series):
                Target variable series.

            class_names (Optional[List[str]], optional):
                Class names associated with the target variable.
                Defaults to None.

            is_regressor (bool, optional):
                Whether the model is a regression model.
                Defaults to False.

        Returns:
            pd.Series: Preprocessed target variable.
        """
        if class_names is None:
            return preprocess_labels(y, class_names, is_regressor)
        else:
            return y

    @abstractmethod
    def get_X(self, *args: Any) -> Any:
        """
        Get the feature data.
        """
        pass

    @abstractmethod
    def get_y(self, *args: Any) -> Any:
        """
        Get the target data.
        """
        pass


class DependenceDataManager(BaseDataManager):
    """
    Data manager for feature dependence analysis.

    This class handles data preparation for analyzing how model predictions
    depend on specific features. It validates input data, applies preprocessing,
    and ensures proper formatting of features and target variables.

    Attributes:
        X (pd.DataFrame): Preprocessed feature data.
        y (pd.Series): Preprocessed target variable.
        column_names (List[str]): Names of features in X.
        class_names (List[str]): Names of classes in y.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        verbose: Literal[0, 1, 2] = 0,
    ) -> None:
        """
        Initialize the DependenceDataManager.

        Args:
            X (pd.DataFrame):
                Feature data.

            y (pd.Series):
                Target variable.

            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to analyze.

            column_names (Optional[List[str]], optional):
                Custom column names for features. If None, uses existing column names.
                Defaults to None.

            class_names (Optional[List[str]], optional):
                Custom class names for target variable. If None, generates appropriate names.
                Defaults to None.

            verbose (Literal[0, 1, 2], optional):
                Controls verbosity of warning and information messages.
                Defaults to 0.
        """
        self._validate_X_and_y(X, y, verbose)
        super().__init__(model, verbose)

        # Transform data if model is a Pipeline
        if self.preprocessor is not None:
            column_names: List[str] = X.columns if column_names is None else column_names
            X: pd.DataFrame = self.preprocessor.transform(X)

        # Preprocess input data
        self.X, self.column_names = self._preprocess_features_and_column_names(
            X, column_names, verbose, self.preprocessor
        )
        self.y: pd.Series = self._preprocess_labels(y, class_names, self.is_regressor)
        self.class_names: List[str] = self._preprocess_class_names(self.y, class_names, self.is_regressor)

    def get_X(self, split_selection: Literal["full", "train", "test"]) -> pd.DataFrame:
        """
        Get the feature data.
        """
        if split_selection == "full":
            return self.X
        else:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")

    def get_y(self, split_selection: Literal["full", "train", "test"]) -> pd.Series:
        """
        Get the target data.
        """
        if split_selection == "full":
            return self.y
        else:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")

    @staticmethod
    def _validate_X_and_y(X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate feature and target data consistency.

        Checks that X and y have compatible dimensions and indexes.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target variable.

        Raises:
            ValueError: If X and y have different number of samples.
        """
        _validate_index_of_X_and_y(X, y)
        _validate_sample_nr_X_and_y(X, y)


class RFEDataManager(DependenceDataManager):
    """
    Data manager for Recursive Feature Elimination (RFE) analysis.

    This class extends DependenceDataManager with additional functionality
    for handling sample weights and cross-validation parameters needed for RFE.

    Attributes:
        Inherits all attributes from DependenceDataManager.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseModel,
        sample_weight: Optional[pd.Series] = None,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        verbose: Literal[0, 1, 2] = 0,
    ) -> None:
        """
        Initialize the RFEDataManager.

        Args:
            X (pd.DataFrame):
                Feature data.

            y (pd.Series):
                Target variable.

            model (BaseModel):
                The model to use for RFE.

            sample_weight (Optional[pd.Series], optional):
                Sample weights for weighted learning.
                Defaults to None.

            column_names (Optional[List[str]], optional):
                Custom column names for features. If None, uses existing column names.
                Defaults to None.

            class_names (Optional[List[str]], optional):
                Custom class names for target variable. If None, generates appropriate names.
                Defaults to None.

            verbose (Literal[0, 1, 2], optional):
                Controls verbosity of warning and information messages.
                Defaults to 0.
        """
        # Validate input data
        self._validate_X_y_and_sample_weight(X, y, sample_weight, verbose)

        super().__init__(X, y, model, column_names, class_names, verbose)

    @staticmethod
    def _validate_X_and_y(
        X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series], verbose: Literal[0, 1, 2] = 0
    ) -> None:
        """
        Validate feature data, target data, and sample weights.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target variable.
            sample_weight (Optional[pd.Series]): Sample weights.
            verbose (Literal[0, 1, 2]): Verbosity level.

        Raises:
            ValueError: If validation fails.
        """
        _validate_sample_weight(sample_weight, X, verbose)
        _validate_index_of_X_and_y(X, y)
        _validate_sample_nr_X_and_y(X, y)

    def calculate_cv_parameter(
        self,
        cv: Union[
            Iterable,
            int,
            BaseShuffleSplit,
            BaseCrossValidator,
        ],
    ) -> BaseCrossValidator:
        """
        Calculate cross-validation parameter for RFE.

        This method configures cross-validation based on the input cv parameter
        and whether the model is a classifier or regressor.

        Args:
            cv (Union[Iterable, int, BaseShuffleSplit, BaseCrossValidator]):
                Cross-validation strategy specification.

        Returns:
            BaseCrossValidator: Configured cross-validator object.
        """
        return check_cv(cv, self.y, classifier=is_classifier(self.model))


class ModelInterpreterDataManager(BaseDataManager):
    """
    Data manager for model interpretation with separate train and test sets.

    This class handles data preparation for models that need to be trained and evaluated
    on separate datasets, with consistent preprocessing applied to both sets.

    Attributes:
        X_train (pd.DataFrame): Preprocessed training features.
        X_test (pd.DataFrame): Preprocessed test features.
        y_train (pd.Series): Preprocessed training target.
        y_test (pd.Series): Preprocessed test target.
        column_names (List[str]): Names of features.
        class_names (List[str]): Names of target classes.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model: Union[BaseEstimator, BaseSearchCV, Pipeline],
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        verbose: Literal[0, 1, 2] = 0,
    ) -> None:
        """
        Initialize the ModelInterpreterDataManager.

        Args:
            X_train (pd.DataFrame):
                Training feature data.

            X_test (pd.DataFrame):
                Test feature data.

            y_train (pd.Series):
                Training target variable.

            y_test (pd.Series):
                Test target variable.

            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to interpret.

            column_names (Optional[List[str]], optional):
                Custom column names for features. If None, uses existing column names.
                Defaults to None.

            class_names (Optional[List[str]], optional):
                Custom class names for target variable. If None, generates appropriate names.
                Defaults to None.

            verbose (Literal[0, 1, 2], optional):
                Controls verbosity of warning and information messages.
                Defaults to 0.
        """
        super().__init__(model, verbose)
        self._validate_X_and_y(X_train, y_train, X_test, y_test, verbose)
        # Transform data if model is a Pipeline
        if self.preprocessor is not None:
            column_names = X_train.columns if column_names is None else column_names
            X_train = self.preprocessor.transform(X_train)
            X_test = self.preprocessor.transform(X_test)

        # Preprocess input data
        self.X_train, self.column_names = self._preprocess_features_and_column_names(
            X_train, X_name="X_train", column_names=column_names, verbose=verbose, preprocessor=self.preprocessor
        )
        self.X_test, _ = self._preprocess_features_and_column_names(
            X_test, X_name="X_test", column_names=column_names, verbose=verbose, preprocessor=self.preprocessor
        )
        self.y_train: pd.Series = self._preprocess_labels(y_train, class_names, self.is_regressor)
        self.y_test: pd.Series = self._preprocess_labels(y_test, class_names, self.is_regressor)
        self.class_names: List[str] = self._preprocess_class_names(pd.concat([self.y_train, self.y_test]), class_names)

    def get_X(self, split_selection: Literal["full", "train", "test"]) -> pd.DataFrame:
        """
        Get the feature data.
        """
        if split_selection == "full":
            return pd.concat([self.X_train, self.X_test])
        elif split_selection == "train":
            return self.X_train
        elif split_selection == "test":
            return self.X_test
        else:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")

    def get_y(self, split_selection: Literal["full", "train", "test"]) -> pd.Series:
        """
        Get the target data.
        """
        if split_selection == "full":
            return pd.concat([self.y_train, self.y_test])
        elif split_selection == "train":
            return self.y_train
        elif split_selection == "test":
            return self.y_test
        else:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")

    def convert_to_dependence_data_manager(self) -> DependenceDataManager:
        return DependenceDataManager(
            X=pd.concat([self.X_train, self.X_test]),
            y=pd.concat([self.y_train, self.y_test]),
            model=self.model,
            column_names=self.column_names,
            class_names=self.class_names,
            verbose=self.verbose,
        )

    @staticmethod
    def _validate_X_and_y(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: Literal[0, 1, 2] = 0,
    ) -> None:
        """
        Validate training and test data for consistency.

        Ensures that training and test datasets have compatible dimensions,
        features, and classes.

        Args:
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target variable.
            X_test (pd.DataFrame): Test feature data.
            y_test (pd.Series): Test target variable.
            verbose (Literal[0, 1, 2]): Verbosity level.

        Raises:
            ValueError: If validation fails.
        """
        _validate_index_of_X_and_y(X_train, y_train, verbose)
        _validate_index_of_X_and_y(X_test, y_test, verbose)
        _validate_sample_nr_X_and_y(X_train, y_train)
        _validate_sample_nr_X_and_y(X_test, y_test)
        _validate_feature_nr_X1_and_X2(X_train, X_test)
        _validate_classes_nr_y_n_and_y_m(y_train, y_test, verbose)


class ImportanceDataManager(BaseDataManager):
    """
    Data manager for feature importance analysis between two datasets.

    This class prepares data for analyzing the differences between two datasets
    by creating a binary classification problem and splitting data into train/test sets.

    Attributes:
        X_train (pd.DataFrame): Training features for the binary classification.
        X_test (pd.DataFrame): Test features for the binary classification.
        y_train (pd.Series): Training target (binary labels).
        y_test (pd.Series): Test target (binary labels).
        column_names (List[str]): Names of features.
        class_names (List[str]): Names for the two classes representing the datasets.
    """

    def __init__(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        model: BaseModel,
        X_test_size: float = 0.25,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the ImportanceDataManager.

        Args:
            X1 (pd.DataFrame):
                First dataset features.

            X2 (pd.DataFrame):
                Second dataset features.

            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
                The model to use for importance analysis.

            X_test_size (float, optional):
                Proportion of data to use for testing.
                Defaults to 0.25.

            column_names (Optional[List[str]], optional):
                Custom column names for features. If None, uses existing column names.
                Defaults to None.

            class_names (Optional[List[str]], optional):
                Custom names for the two classes representing the datasets.
                Defaults to None.

            verbose (Literal[0, 1, 2], optional):
                Controls verbosity of warning and information messages.
                Defaults to 0.

            random_state (Optional[int], optional):
                Random seed for reproducibility.
                Defaults to None.
        """
        self._validate_X(X1, X2)
        super().__init__(model, verbose)

        # Transform data if model is a Pipeline
        if self.model.preprocessor is not None:
            column_names = X1.columns if column_names is None else column_names
            X1 = self.model.preprocessor.transform(X1)
            X2 = self.model.preprocessor.transform(X2)

        # Preprocess input data
        X1, self.column_names = self._preprocess_features_and_column_names(
            X1, X_name="X1", column_names=column_names, verbose=verbose, preprocessor=self.model.preprocessor
        )
        X2, _ = self._preprocess_features_and_column_names(
            X2, X_name="X2", column_names=column_names, verbose=verbose, preprocessor=self.model.preprocessor
        )

        # Create binary dataset & split it into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test, self.class_names = (
            self._create_stratified_binary_classification_train_test_split(
                X1, X2, X_test_size, class_names, random_state
            )
        )

    def get_X(self, split_selection: Literal["full", "train", "test"]) -> pd.DataFrame:
        """
        Get the feature data.
        """
        if split_selection == "full":
            return pd.concat([self.X_train, self.X_test])
        elif split_selection == "train":
            return self.X_train
        elif split_selection == "test":
            return self.X_test
        else:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")

    def get_y(self, split_selection: Literal["full", "train", "test"]) -> pd.Series:
        """
        Get the target data.
        """
        if split_selection == "full":
            return pd.concat([self.y_train, self.y_test])
        elif split_selection == "train":
            return self.y_train
        elif split_selection == "test":
            return self.y_test
        else:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")

    @staticmethod
    def _validate_X(X1: pd.DataFrame, X2: pd.DataFrame) -> None:
        """
        Validate that two datasets have the same number of features.

        Args:
            X1 (pd.DataFrame): First dataset.
            X2 (pd.DataFrame): Second dataset.

        Raises:
            ValueError: If X1 and X2 have different number of features.
        """
        _validate_feature_nr_X1_and_X2(X1, X2)

    @staticmethod
    def _create_stratified_binary_classification_train_test_split(
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        X_test_size: float = 0.25,
        class_names: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Create a binary classification dataset and split it into train/test sets.

        This method combines two datasets into one, assigns binary labels (0 for X1, 1 for X2),
        and creates a stratified train/test split.

        Args:
            X1 (pd.DataFrame): First dataset (will be labeled as class 0).
            X2 (pd.DataFrame): Second dataset (will be labeled as class 1).
            X_test_size (float, optional): Proportion of data to use for testing. Defaults to 0.25.
            class_names (Optional[List[str]], optional): Custom names for the two classes. Defaults to None.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
                - X_train: Training features
                - X_test: Test features
                - y_train: Training binary labels
                - y_test: Test binary labels
                - class_names: Names for the two classes
        """
        X, y = pd.concat([X1, X2], axis=0), pd.concat([pd.Series(0, index=X1.index), pd.Series(1, index=X2.index)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=X_test_size, random_state=random_state, shuffle=True, stratify=y
        )
        y: pd.Series = ImportanceDataManager._preprocess_labels(y, class_names, False)
        class_names: List[str] = ImportanceDataManager._preprocess_class_names(y, class_names, False)

        return X_train, X_test, y_train, y_test, class_names


class ModelInterpreterDependenceDataManager(DependenceDataManager):
    """
    Data manager for feature dependence analysis.

    This class handles data preparation for analyzing how model predictions
    depend on specific features. It validates input data, applies preprocessing,
    and ensures proper formatting of features and target variables.

    Attributes:
        X (pd.DataFrame): Preprocessed feature data.
        y (pd.Series): Preprocessed target variable.
        column_names (List[str]): Names of features in X.
        class_names (List[str]): Names of classes in y.
    """

    def __init__(
        self,
        model: BaseModel,
        data_manager: ModelInterpreterDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
    ) -> None:
        """
        Initialize the ModelInterpreterDependenceDataManager.

        Args:
            model (BaseModel):
                Model to use for dependence analysis.

            data_manager (ModelInterpreterDataManager):
                Data manager to convert to DependenceDataManager.

            split_selection (Literal["full", "train", "test"], optional):
                Split selection for the data.
                Defaults to "test".
        """
        # Allow SHAP cache to work when using split_selection, since we are
        # using the same data manager for both train and test for multiple classes,
        # namely ShapModelInterpreter with ShapInterpreterDependencePlotter when used
        # in conjunction. Otherwise ShapDependencePlotter is used independently, which
        # does not require split_selection, but works with X and y naming.
        if split_selection not in ["full", "train", "test"]:
            raise ValueError(f"Invalid split_selection: {split_selection}, only full, train and test are supported.")
        if split_selection == "full":
            self.X: pd.DataFrame = pd.concat([data_manager.X_train, data_manager.X_test])
            self.y: pd.Series = pd.concat([data_manager.y_train, data_manager.y_test])
        elif split_selection == "train":
            self.X = self.X_train = data_manager.X_train
            self.y = self.y_train = data_manager.y_train
        elif split_selection == "test":
            self.X = self.X_test = data_manager.X_test
            self.y = self.y_test = data_manager.y_test

        # Validate data
        self._validate_X_and_y(self.X, self.y, self.verbose)

        # Set attributes
        self.model: BaseModel = model
        self.verbose: Literal[0, 1, 2] = model.verbose
        self.column_names: List[str] = data_manager.column_names
        self.class_names: List[str] = data_manager.class_names


def _validate_sample_weight(
    sample_weight: Optional[pd.Series], X: pd.DataFrame, verbose: Literal[0, 1, 2] = 0
) -> Optional[pd.Series]:
    """
    Validate sample weights and ensure proper formatting.

    Args:
        sample_weight (Optional[pd.Series]): Sample weights to validate.
        X (pd.DataFrame): Feature data that sample weights will be applied to.
        verbose (Literal[0, 1, 2], optional): Verbosity level. Defaults to 0.

    Returns:
        Optional[pd.Series]: Validated sample weights or None if not provided.
    """
    if sample_weight is not None:
        if verbose > 0:
            warnings.warn("sample_weight is passed only to the fit method of the model, not the evaluation metrics.")
        sample_weight = assure_pandas_series(sample_weight, index=X.index)
    return sample_weight


def _validate_index_of_X_and_y(X: pd.DataFrame, y: pd.Series, verbose: Literal[0, 1, 2] = 0) -> None:
    """
    Validate that X and y have the same index.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.
        verbose (Literal[0, 1, 2], optional): Verbosity level. Defaults to 0.
    """
    # Warning if index is not the same
    if verbose > 0:
        if not X.index.equals(y.index):
            warnings.warn("X and y have different index.")


def _validate_sample_nr_X_and_y(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validate that X and y have the same number of samples.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.

    Raises:
        ValueError: If X and y have different number of samples.
    """
    # Error if number of samples is not the same
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different number of samples.")


def _validate_feature_nr_X1_and_X2(X1: pd.DataFrame, X2: pd.DataFrame) -> None:
    """
    Validate that two datasets have the same number of features.

    Args:
        X1 (pd.DataFrame): First dataset.
        X2 (pd.DataFrame): Second dataset.

    Raises:
        ValueError: If X1 and X2 have different number of features.
    """
    # Error if number of features is not the same
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 have different number of features.")


def _validate_classes_nr_y_n_and_y_m(y_n: pd.Series, y_m: pd.Series, verbose: Literal[0, 1, 2] = 0) -> None:
    """
    Validate that two target variables have the same set of classes.

    Args:
        y_n (pd.Series): First target variable.
        y_m (pd.Series): Second target variable.
        verbose (Literal[0, 1, 2], optional): Verbosity level. Defaults to 0.
    """
    if verbose > 0:
        # Warning if classes are not the same
        if y_n.nunique() != y_m.nunique():
            warnings.warn(f"y_n and y_m classes are not the same: {y_n.unique()} and {y_m.unique()}")

        # Warning if either one is not a subset of the other, thus covering a different set of classes
        if not y_n.isin(y_m).all() or not y_m.isin(y_n).all():
            warnings.warn(f"y_n and y_m cover different classes: {y_n.unique()} and {y_m.unique()}")
