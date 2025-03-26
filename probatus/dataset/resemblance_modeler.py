import warnings
from typing import Any, List, Optional, Tuple, Union, cast, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from distutils.version import LooseVersion
from shap import Explanation
from shap.plots import bar, beeswarm, waterfall

from sklearn.pipeline import Pipeline

from probatus.core import BaseFitComputePlotClass
from probatus.utils import (
    preprocess_data,
    preprocess_labels,
    get_single_scorer,
    Scorer,
    calculate_shap_importance,
    calculate_shap_explanation,
    shap_explanation_to_shap_df,
    extract_shap_multiclass_params,
    get_pipeline_preprocessor_and_estimator,
)


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
        if isinstance(model, Pipeline):
            self.pipeline, self.preprocessor = get_pipeline_preprocessor_and_estimator(model)
        else:
            self.pipeline = None
            self.preprocessor = None
        self.model = model
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
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = ["First Sample", "Second Sample"]

        # Transform data if model is a Pipeline
        if self.pipeline is not None:
            column_names = X1.columns if column_names is None else column_names
            X1 = self.pipeline.transform(X1)
            X2 = self.pipeline.transform(X2)

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
        scoring: Union[str, Scorer] = "roc_auc",
        test_prc: float = 0.25,
        n_jobs: int = 1,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initializes the PermutationImportanceResemblance model.

        Args:
            model (BaseEstimator):
                A machine learning classifier used to distinguish between samples.
                Must implement `fit()` and either `predict()` or `predict_proba()`.

            iterations (int, optional):
                Number of iterations for permutation importance calculation.
                Higher values improve stability but increase computation time.
                Defaults to `100`.

            scoring (str, optional):
                Metric used to evaluate model performance.
                Can be a string referring to a scikit-learn classification metric
                (see: https://scikit-learn.org/stable/modules/model_evaluation.html)
                or a `probatus.utils.Scorer` object for custom scoring.
                Defaults to `"roc_auc"`.

            test_prc (float, optional):
                Fraction of data used for testing, in the range `(0, 1]`.
                Defaults to `0.25`.

            n_jobs (int, optional):
                Number of parallel jobs to use. Set to `-1` to utilize all available cores.
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
        self.iterations_results = pd.DataFrame(
            {"feature": pd.Series(dtype="object"), "importance": pd.Series(dtype="float64")}
        )
        self.iterations_columns = self.iterations_results.columns

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
        self.report_df = pd.DataFrame(index=self.column_names, columns=self.report_columns, dtype=float)

        # Process results for each feature
        for feature_index, feature_name in enumerate(self.column_names):
            # Store summary statistics
            self.report_df.loc[feature_name, "mean_importance"] = permutation_result["importances_mean"][feature_index]
            self.report_df.loc[feature_name, "std_importance"] = permutation_result["importances_std"][feature_index]

            # Store individual iteration results for visualization
            feature_iterations = pd.DataFrame(
                {
                    "feature": np.repeat(feature_name, self.iterations),
                    "importance": permutation_result["importances"][feature_index, :].reshape((self.iterations,)),
                }
            )

            # Append to overall results
            if not feature_iterations.empty:
                self.iterations_results = pd.concat([self.iterations_results, feature_iterations])

        # Sort features by importance (descending)
        self.report_df.sort_values(by="mean_importance", ascending=False, inplace=True)

        return self

    def plot(self, top_n: Optional[int] = None, show: bool = False, **plot_kwargs: Any) -> Figure:
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
                If True, calls plt.show()
                If False, returns the figure without displaying (default)
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

        # Setup plotting environment
        was_interactive = plt.isinteractive()
        plt.ioff()

        feature_report: pd.DataFrame = self.compute(return_scores=False)
        sorted_features = feature_report["mean_importance"].sort_values(ascending=True).index.values

        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[-top_n:]

        # Set default figure size if not provided
        if "figsize" not in plot_kwargs:
            num_features = len(sorted_features)
            # Adjust height based on number of features (minimum 6 inches)
            height = max(6, 0.5 * num_features)
            plot_kwargs["figsize"] = (10, height)

        # Apply SHAP-like style
        plt.style.use("default")  # Reset to default style first

        fig, ax = plt.subplots(**plot_kwargs)

        # Set light gray background with white grid
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("white")

        # Create boxplots for each feature
        for position, feature in enumerate(sorted_features):
            # Get importance values for this feature
            feature_values = self.iterations_results[self.iterations_results["feature"] == feature]["importance"]

            # TODO: Remove this once we drop support for matplotlib < 3.10
            # Create horizontal boxplot
            if LooseVersion(matplotlib.__version__) >= LooseVersion("3.10"):
                # Use orientation parameter for matplotlib 3.10+
                box = ax.boxplot(
                    feature_values,
                    positions=[position],
                    orientation="horizontal",
                    patch_artist=True,  # Fill boxplots
                )
            else:
                # Use vert=False for older matplotlib versions
                box = ax.boxplot(
                    feature_values,
                    positions=[position],
                    vert=False,
                    patch_artist=True,  # Fill boxplots
                )

            # Style the boxplots with SHAP-like colors
            for patch in box["boxes"]:
                patch.set_facecolor("#1E88E5")
                patch.set_alpha(0.6)
            for median in box["medians"]:
                median.set_color("#ff0051")
                median.set_linewidth(2)

        # Add subtle grid lines
        ax.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

        # Set custom tick parameters
        ax.tick_params(axis="both", which="major", labelsize=10)

        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel(self.plot_x_label, fontsize=11, fontweight="bold")
        ax.set_ylabel(self.plot_y_label, fontsize=11, fontweight="bold")
        ax.set_title(self.plot_title, fontsize=13, fontweight="bold", pad=15)

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

        # Add a thin border
        for spine in ax.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.8)

        # Adjust figure margins to make room for annotations
        plt.subplots_adjust(bottom=0.2)

        # Adjust layout to make sure everything fits
        plt.tight_layout()

        # Finalize and handle display
        plt.tight_layout()
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)

        # Restore interactive state
        if was_interactive:
            plt.ion()

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
        scoring: Union[str, Scorer] = "roc_auc",
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

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
                - Default is `0`.

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

            **shap_kwargs (Any):
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values multi-classification conversion - parameters like 'class_selection', 'multi-class_aggregation', and 'weights'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multi-class models.

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

        # Split arguments for multi-classification
        multi_class_kwargs, shap_kwargs = extract_shap_multiclass_params(shap_kwargs)

        # Calculate SHAP values for test set
        # SHAP values explain each feature's contribution to model predictions
        self.shap_explanation_test = calculate_shap_explanation(
            self.model,
            self.X_test,
            return_explainer=False,
            verbose=self.verbose,
            random_state=self.random_state,
            **shap_kwargs,
        )
        self.shap_values_test = shap_explanation_to_shap_df(
            shap_explanation=self.shap_explanation_test,
            model=self.model,
            X=self.X_test,
            **multi_class_kwargs,
        )

        # Calculate feature importance from SHAP values
        shap_df = pd.DataFrame(self.shap_values_test, columns=self.column_names)
        self.report_df = calculate_shap_importance(shap_df, self.column_names)

        return self

    def plot(
        self,
        plot: Union[Literal["bar", "beeswarm"], int] = "bar",
        results_text: str = None,
        plot_title: str = None,
        show: bool = False,
        **plot_kwargs: Any,
    ) -> Figure:
        """
        Create a SHAP plot to visualize feature importance.

        This method uses SHAP's modern visualization tools to create different types of plots
        showing the impact of features on model predictions. The plot includes both
        the magnitude and direction of each feature's effect.

        Args:
            plot (Union[Literal["bar", "beeswarm"], int], optional): Type of SHAP plot.
                Options:
                - "bar": Bar chart showing average absolute SHAP values (default)
                - "beeswarm": Beeswarm plot showing distribution of SHAP values and feature values
                - int: Sample index to use for a waterfall plot of that specific sample's prediction.
                Defaults to `"bar"`.

            results_text (str): Text describing performance metrics to display below the plot.

            plot_title (str): Title to display on the plot.

            show (bool, optional): Whether to display the plot immediately.
                If True, calls plt.show()
                If False, returns the figure without displaying (default)

            **plot_kwargs (Any): Additional keyword arguments passed to the respective SHAP plot function.
                Common options include:
                - max_display: int, maximum number of features to show
                - color: str or color map, color scheme for the plot
                - order: specific ordering for features

        Returns:
            matplotlib.figure.Figure: The created figure object.
                Can be used for further customization or saving to file.

        Raises:
            ValueError: If plot is not one of "bar", "beeswarm", or an integer.
            ValueError: If required data is missing or improperly formatted.
        """
        self._check_if_fitted()

        # Setup and validate
        was_interactive = plt.isinteractive()
        plt.ioff()

        if isinstance(plot, str) and plot not in ["bar", "beeswarm"]:
            raise ValueError("plot must be one of 'bar', 'beeswarm', or an integer sample index")

        # Get feature names and prepare plot parameters
        feature_names = self.column_names or (
            self.X_test.columns.tolist() if isinstance(self.X_test, pd.DataFrame) else None
        )
        actual_plot_type = "waterfall" if isinstance(plot, (int, np.int64)) else plot
        plot_kwargs["max_display"] = min(
            plot_kwargs.get("max_display", 20), len(feature_names) if feature_names else 20
        )

        # Create visualization
        explanation = self._create_explanation(self.shap_values_test, self.X_test, feature_names, plot)
        fig = plt.figure(figsize=(10, max(6, 0.4 * plot_kwargs.get("max_display"))))
        plot_funcs = {"bar": bar, "beeswarm": beeswarm, "waterfall": waterfall}
        plot_funcs[actual_plot_type](explanation, show=False, **plot_kwargs)

        # Style and finalize
        self._style_plot(fig, plot, plot_title, results_text)
        plt.tight_layout()

        # Handle display and restore state
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)

        if was_interactive:
            plt.ion()

        return fig

    @staticmethod
    def _create_explanation(
        shap_values: pd.DataFrame,
        X: pd.DataFrame,
        feature_names: List[str],
        plot_type: Union[Literal["bar", "beeswarm"], int],
    ) -> Explanation:
        """
        Convert input data to a SHAP Explanation object based on input type and plot type.

        This helper function handles different input data types and formats them for
        compatibility with SHAP's visualization functions.

        Args:
            shap_values (pd.DataFrame): SHAP values as a pandas DataFrame.
            X (pd.DataFrame): Feature values for the test data used to generate SHAP values.
            feature_names (List[str]): List of feature names to include in the Explanation.
            plot_type (Union[Literal["bar", "beeswarm"], int]): Determines visualization type:
                - "bar"/"beeswarm": Use all samples for global explanations
                - int: Extract specific sample at that index for waterfall plot

        Returns:
            Explanation: A properly formatted SHAP Explanation object ready for visualization.
        """
        # For waterfall plots, extract specific sample
        if isinstance(plot_type, (int, np.int64)):
            sample_idx = int(plot_type)
            is_df = isinstance(shap_values, pd.DataFrame)
            values = shap_values.iloc[sample_idx].values if is_df else shap_values[sample_idx]
            data = X.iloc[sample_idx].values if isinstance(X, pd.DataFrame) else X[sample_idx]
            return Explanation(values=values, base_values=0.0, data=data, feature_names=feature_names)

        # DataFrame/array input for bar/beeswarm plots
        values = shap_values.values if isinstance(shap_values, pd.DataFrame) else shap_values
        data = X.values if isinstance(X, pd.DataFrame) else X
        return Explanation(values=values, data=data, feature_names=feature_names)

    @staticmethod
    def _style_plot(
        fig: Figure, plot_type: Union[Literal["bar", "beeswarm"], int], plot_title: str, results_text: str
    ) -> None:
        """
        Apply consistent styling to SHAP visualization figures.

        Enhances the default SHAP plots with custom styling for improved readability
        and presentation.

        Args:
            fig (Figure): Matplotlib figure object to style.
            plot_type (Union[Literal["bar", "beeswarm"], int]): Type of SHAP plot.
            plot_title (str): Title text to display above the plot.
            results_text (str): Performance metrics or other descriptive text
                to display below the plot.

        Returns:
            None: Modifies the figure in-place.
        """
        ax = plt.gca()

        # Apply styling
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

        if isinstance(plot_type, (int, np.int64)):
            plot_title = plot_title if plot_title else f"SHAP Waterfall Plot (Sample #{plot_type})"
        elif plot_type == "beeswarm":
            plot_title = plot_title if plot_title else "SHAP Feature Importance (Beeswarm Plot)"
        else:
            plot_title = plot_title if plot_title else "SHAP Feature Importance (Bar Plot)"

        plt.suptitle(plot_title, fontsize=13, fontweight="bold", y=1.02)
        if results_text:
            plt.figtext(
                0.5,
                -0.05,
                results_text,
                ha="center",
                fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5, "edgecolor": "lightgray"},
            )

        # Style text and borders
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(10)
        for spine in ax.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.8)
