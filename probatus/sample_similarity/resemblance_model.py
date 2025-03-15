import warnings
from typing import Any, List, Optional, Tuple, Union, cast, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from shap import summary_plot
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from probatus.utils import BaseFitComputePlotClass, preprocess_data, preprocess_labels, get_single_scorer
from probatus.utils.shap_helpers import calculate_shap_importance, shap_calc


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
        model: BaseEstimator,
        scoring: str = "roc_auc",
        test_prc: float = 0.25,
        n_jobs: int = 1,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the BaseResemblanceModel.

        Args:
            model: Regression or classification model or pipeline.
                Must implement fit() and predict() or predict_proba() methods.

            scoring: Metric for model performance evaluation.
                Can be a string matching sklearn's classification metrics
                (see: https://scikit-learn.org/stable/modules/model_evaluation.html)
                or a probatus.utils.Scorer object for custom metrics.
                'roc_auc' is recommended for this class.

            test_prc: Percentage of data used for testing the model (default: 0.25).

            n_jobs: Number of parallel jobs to run.
                Set to -1 to use all available cores (default: 1).

            verbose: Controls output verbosity:
                0 - No output or warnings
                1 - Only important warnings
                2 - All prints and warnings

            random_state: Random seed for reproducibility.
                Set to an integer for reproducible results or None for non-reproducible behavior.
        """
        self.model = model
        self.test_prc = test_prc
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.scorer = get_single_scorer(scoring)
        self.fitted = False
        self.report: Optional[pd.DataFrame] = None

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
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = ["First Sample", "Second Sample"]

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
            f"Train {self.scorer.metric_name}: {self.train_score},\n"
            f"Test {self.scorer.metric_name}: {self.test_score}."
        )
        if self.verbose > 1:
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

    def _get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Return the data splits used to train the Resemblance model.

        This method provides access to the training and test data splits created during model fitting.
        The data is split to maintain class balance through stratification.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                A tuple containing:
                - X_train (pd.DataFrame): Training features of shape (n_train_samples, n_features)
                - X_test (pd.DataFrame): Test features of shape (n_test_samples, n_features)
                - y_train (pd.Series): Training labels of shape (n_train_samples,)
                - y_test (pd.Series): Test labels of shape (n_test_samples,)

        Raises:
            ValueError: If the model has not been fitted yet. Call fit() before accessing data splits.
        """
        self._check_if_fitted()
        return (
            cast(pd.DataFrame, self.X_train),
            cast(pd.DataFrame, self.X_test),
            cast(pd.Series, self.y_train),
            cast(pd.Series, self.y_test),
        )

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
            return self.report, cast(float, self.train_score), cast(float, self.test_score)
        else:
            return self.report

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


class PermutationImportanceResemblance(BaseResemblanceModel):
    """
    Resemblance model using permutation importance to identify key distinguishing features.

    This model analyzes the similarity between two samples by:
    1. Labeling each sample (0 for first sample, 1 for second sample)
    2. Training a model to distinguish between the samples
    3. Using permutation importance to identify which features are most important for distinguishing

    Interpretation:
    - If the model achieves a test score significantly different from 0.5, the samples are distinguishable
    - Features with high permutation importance contribute most to the differences between samples
    - These features likely have different distributions between the two samples

    Examples:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.sample_similarity import PermutationImportanceResemblance
    import pandas as pd

    # Create two slightly different datasets
    X1_array, _ = make_classification(n_samples=100, n_features=5)
    X2_array, _ = make_classification(n_samples=100, n_features=5, shift=0.5)

    # Convert to pandas DataFrames
    feature_names = [f'feature_{i}' for i in range(5)]
    X1 = pd.DataFrame(X1_array, columns=feature_names)
    X2 = pd.DataFrame(X2_array, columns=feature_names)

    # Initialize and fit the model
    model = RandomForestClassifier(max_depth=2)
    perm = PermutationImportanceResemblance(model)
    feature_importance = perm.fit_compute(X1, X2)

    # Visualize the results
    perm.plot()
    ```
    <img src="../img/sample_similarity_permutation_importance.png" width="500" />
    """

    def __init__(
        self,
        model: BaseEstimator,
        iterations: int = 100,
        scoring: str = "roc_auc",
        test_prc: float = 0.25,
        n_jobs: int = 1,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the PermutationImportanceResemblance model.

        Args:
            model: Machine learning model (classifier) to distinguish between samples.
                Must implement fit() and predict() or predict_proba() methods.

            iterations: Number of iterations for permutation importance calculation.
                Higher values give more stable results but take longer (default: 100).

            scoring: Metric for model performance evaluation.
                Can be a string matching sklearn's classification metrics
                or a probatus.utils.Scorer object for custom metrics.
                'roc_auc' is recommended for this class.

            test_prc: Percentage of data used for testing (default: 0.25).

            n_jobs: Number of parallel jobs to run.
                Set to -1 to use all available cores (default: 1).

            verbose: Controls output verbosity:
                0 - No output or warnings
                1 - Only important warnings
                2 - All prints and warnings

            random_state: Random seed for reproducibility.
                Set to an integer for reproducible results.
        """
        super().__init__(
            model=model,
            scoring=scoring,
            test_prc=test_prc,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

        self.iterations = iterations

        # Initialize dataframe to store iteration results
        self.iterations_columns = ["feature", "importance"]
        self.iterations_results = pd.DataFrame(columns=self.iterations_columns)

        # Set plot labels
        self.plot_x_label = "Permutation Feature Importance"
        self.plot_y_label = "Feature Name"
        self.plot_title = "Permutation Feature Importance of Resemblance Model"

    def fit(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> "PermutationImportanceResemblance":
        """
        Fit the model and calculate permutation importance.

        This method extends the base class fit method by adding permutation importance calculation.
        After fitting the model on the training data, it evaluates feature importance by measuring
        how much model performance decreases when each feature is randomly shuffled.

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
            PermutationImportanceResemblance: The fitted model instance with calculated importance values.

        Raises:
            ValueError: If input data dimensions don't match.
            ValueError: If column_names length doesn't match number of features.
            ValueError: If class_names is provided but not of length 2.
            Warning: If train score is significantly higher than test score,
                    indicating potential overfitting.
        """
        # Call parent class fit method to prepare data and train model
        super().fit(X1=X1, X2=X2, column_names=column_names, class_names=class_names)

        # Calculate permutation importance
        # This measures how model performance decreases when a feature is randomly shuffled
        permutation_result = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            scoring=self.scorer.scorer,
            n_repeats=self.iterations,
            n_jobs=self.n_jobs,
        )

        # Create report dataframe
        self.report_columns = ["mean_importance", "std_importance"]
        self.report = pd.DataFrame(index=self.column_names, columns=self.report_columns, dtype=float)

        # Process results for each feature
        for feature_index, feature_name in enumerate(self.column_names):
            # Store summary statistics
            self.report.loc[feature_name, "mean_importance"] = permutation_result["importances_mean"][feature_index]
            self.report.loc[feature_name, "std_importance"] = permutation_result["importances_std"][feature_index]

            # Store individual iteration results for visualization
            feature_iterations = pd.DataFrame(
                {
                    "feature": np.repeat(feature_name, self.iterations),
                    "importance": permutation_result["importances"][feature_index, :].reshape((self.iterations,)),
                }
            )

            # Append to overall results
            self.iterations_results = pd.concat([self.iterations_results, feature_iterations])

        # Sort features by importance (descending)
        self.report.sort_values(by="mean_importance", ascending=False, inplace=True)

        return self

    def plot(self, top_n: Optional[int] = None, show: bool = True, **plot_kwargs: Any) -> Figure:
        """
        Plot feature importance as boxplots showing the distribution of importance values.

        This method creates a horizontal boxplot visualization where:
        - Each row represents a feature
        - The boxplot shows the distribution of importance values across iterations
        - Features are sorted by mean importance (most important at the top)
        - Performance metrics are annotated below the plot

        Args:
            top_n (Optional[int], optional): Number of top features to include in the plot.
                If None, includes all features.
                If provided, must be positive and <= number of features.
                Features are selected based on mean importance.

            show (bool, optional): Whether to display the plot immediately.
                If True, calls plt.show() (default)
                If False, returns the figure without displaying
                Useful when you want to modify the plot further

            **plot_kwargs (Any): Additional keyword arguments passed to plt.subplots().
                Common options include:
                - figsize: Tuple[float, float] for figure dimensions
                - dpi: Float for figure resolution
                - facecolor: Color for figure background

        Returns:
            matplotlib.figure.Figure: The created figure object.
                Can be used for further customization or saving to file.

        Raises:
            ValueError: If the model has not been fitted yet.
            ValueError: If top_n is provided but not positive.
            ValueError: If top_n is larger than the number of features.
        """
        self._check_if_fitted()
        feature_report: pd.DataFrame = self.compute(return_scores=False)
        sorted_features = feature_report["mean_importance"].sort_values(ascending=True).index.values

        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[-top_n:]

        fig, ax = plt.subplots(**plot_kwargs)

        # Create boxplots for each feature
        for position, feature in enumerate(sorted_features):
            # Get importance values for this feature
            feature_values = self.iterations_results[self.iterations_results["feature"] == feature]["importance"]

            # Create horizontal boxplot
            ax.boxplot(
                feature_values,
                positions=[position],
                vert=False,
            )

        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel(self.plot_x_label)
        ax.set_ylabel(self.plot_y_label)
        ax.set_title(self.plot_title)

        # Add performance metrics annotation
        ax.annotate(
            self.results_text,
            (0, 0),
            (0, -50),
            fontsize=12,
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

        # Show or close the plot
        if show:
            plt.show()
        else:
            # Close plot to improve memory usage when decided not to show
            plt.close(fig)

        return fig


class SHAPImportanceResemblance(BaseResemblanceModel):
    """
    Resemblance model using SHAP values to identify key distinguishing features.

    This model analyzes the similarity between two samples by:
    1. Labeling each sample (0 for first sample, 1 for second sample)
    2. Training a model to distinguish between the samples
    3. Using SHAP (SHapley Additive exPlanations) to identify which features are most important

    Interpretation:
    - If the model achieves a test score significantly different from 0.5, the samples are distinguishable
    - Features with high SHAP importance contribute most to the differences between samples
    - These features likely have different distributions between the two samples

    Note:
    - This class currently works only with tree-based models (like Random Forest, XGBoost)

    Examples:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.sample_similarity import SHAPImportanceResemblance
    import pandas as pd

    # Create two slightly different datasets
    X1_array, _ = make_classification(n_samples=100, n_features=5)
    X2_array, _ = make_classification(n_samples=100, n_features=5, shift=0.5)

    # Convert to pandas DataFrames
    feature_names = [f'feature_{i}' for i in range(5)]
    X1 = pd.DataFrame(X1_array, columns=feature_names)
    X2 = pd.DataFrame(X2_array, columns=feature_names)

    # Initialize and fit the model
    model = RandomForestClassifier(max_depth=2)
    rm = SHAPImportanceResemblance(model)
    feature_importance = rm.fit_compute(X1, X2)

    # Visualize the results
    rm.plot()
    ```

    <img src="../img/sample_similarity_shap_importance.png" width="320" />
    <img src="../img/sample_similarity_shap_summary.png" width="320" />
    """

    def __init__(
        self,
        model: BaseEstimator,
        scoring: str = "roc_auc",
        test_prc: float = 0.25,
        n_jobs: int = 1,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the SHAPImportanceResemblance model.

        Args:
            model: Machine learning model (classifier) to distinguish between samples.
                Must implement fit() and predict() or predict_proba() methods.
                Currently only works with tree-based models.

            scoring: Metric for model performance evaluation.
                Can be a string matching sklearn's classification metrics
                or a probatus.utils.Scorer object for custom metrics.
                'roc_auc' is recommended for this class.

            test_prc: Percentage of data used for testing (default: 0.25).

            n_jobs: Number of parallel jobs to run.
                Set to -1 to use all available cores (default: 1).

            verbose: Controls output verbosity:
                0 - No output or warnings
                1 - Only important warnings
                2 - All prints and warnings

            random_state: Random seed for reproducibility.
                Set to an integer for reproducible results.
        """
        super().__init__(
            model=model,
            scoring=scoring,
            test_prc=test_prc,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

        self.plot_title = "SHAP summary plot"

    def fit(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **shap_kwargs: Any,
    ) -> "SHAPImportanceResemblance":
        """
        Fit the model and calculate SHAP importance values.

        This method extends the base class fit method by adding SHAP value calculation.
        After fitting the model on the training data, it uses SHAP (SHapley Additive exPlanations)
        to explain the model's predictions and determine feature importance.

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

            **shap_kwargs (Any): Additional arguments passed to the SHAP explainer.
                See https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html
                Important options include:
                - approximate: If True, uses faster but less accurate SHAP calculation
                - check_additivity: If False, disables additivity check in SHAP

        Returns:
            SHAPImportanceResemblance: The fitted model instance with calculated SHAP values.

        Raises:
            ValueError: If input data dimensions don't match.
            ValueError: If column_names length doesn't match number of features.
            ValueError: If class_names is provided but not of length 2.
            Warning: If train score is significantly higher than test score,
                    indicating potential overfitting.
            RuntimeError: If model is not tree-based or SHAP calculation fails.
        """
        super().fit(X1=X1, X2=X2, column_names=column_names, class_names=class_names)

        # Calculate SHAP values for test set
        # SHAP values explain each feature's contribution to model predictions
        self.shap_values_test: pd.DataFrame = shap_calc(
            self.model,
            self.X_test,
            return_explainer=False,
            verbose=self.verbose,
            random_state=self.random_state,
            **shap_kwargs,
        )

        # Calculate feature importance from SHAP values
        self.report = calculate_shap_importance(self.shap_values_test, self.column_names)

        return self

    def plot(
        self, plot_type: Literal["bar", "dot", "violin"] = "bar", show: bool = True, **summary_plot_kwargs: Any
    ) -> Figure:
        """
        Create a SHAP summary plot to visualize feature importance.

        This method uses SHAP's visualization tools to create different types of plots
        showing the impact of features on model predictions. The plot includes both
        the magnitude and direction of each feature's effect.

        Args:
            plot_type (Literal["bar", "dot", "violin"], optional): Type of SHAP summary plot.
                Options:
                - "bar": Bar chart showing average absolute SHAP values (default)
                - "dot": Beeswarm plot showing distribution of SHAP values and feature values
                - "violin": Violin plot showing distribution of SHAP values

            show (bool, optional): Whether to display the plot immediately.
                If True, calls plt.show() (default)
                If False, returns the figure without displaying
                Useful when you want to modify the plot further

            **summary_plot_kwargs (Any): Additional keyword arguments passed to shap.summary_plot().
                Common options include:
                - max_display: int, maximum number of features to show
                - plot_size: tuple, figure dimensions
                - color: str/tuple, color of plots
                - alpha: float, transparency of plots

        Returns:
            matplotlib.figure.Figure: The created figure object.
                Can be used for further customization or saving to file.

        Raises:
            ValueError: If the model has not been fitted yet.
            ValueError: If plot_type is not one of "bar", "dot", or "violin".
        """
        self._check_if_fitted()

        # Convert SHAP values to numpy array if they're a DataFrame
        # This is necessary because SHAP's summary_plot expects numpy arrays for dot and violin plots
        shap_values_array = (
            self.shap_values_test.values if isinstance(self.shap_values_test, pd.DataFrame) else self.shap_values_test
        )
        X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test

        # Create SHAP summary plot
        # This creates its own figure and axes internally
        summary_plot(
            shap_values_array,
            X_test_array,
            plot_type=plot_type,
            class_names=self.class_names,
            show=False,  # Don't show yet, we'll add annotations first
            feature_names=self.column_names,
            **summary_plot_kwargs,
        )

        # Get the figure and axes created by summary_plot
        fig, ax = plt.gcf(), plt.gca()

        # Add title
        ax.set_title(self.plot_title)

        # Add performance metrics annotation
        ax.annotate(
            self.results_text,
            (0, 0),
            (0, -50),
            fontsize=12,
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

        # Show or close the plot
        if show:
            plt.show()
        else:
            # Close plot to improve memory usage when decided not to show
            plt.close(fig)

        return fig

    def get_shap_values(self) -> np.ndarray:
        """
        Get the SHAP values calculated for the test set.

        This method provides access to the raw SHAP values computed during model fitting.
        These values can be used for custom analyses or visualizations beyond the
        standard plots provided by the plot() method.

        Returns:
            np.ndarray: Array of SHAP values for the test set.
                Shape: (n_test_samples, n_features)
                Each value represents a feature's contribution to a specific prediction:
                - Positive values push the prediction toward class 1
                - Negative values push the prediction toward class 0
                - Magnitude indicates strength of the effect

        Raises:
            ValueError: If the model has not been fitted yet.
                Call fit() before accessing SHAP values.
        """
        self._check_if_fitted()
        return self.shap_values_test
