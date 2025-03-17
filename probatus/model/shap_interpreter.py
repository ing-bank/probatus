import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shap import summary_plot
from shap import Explanation
from shap.plots import waterfall
from typing import Any, List, Optional, Tuple, Union, Literal
from matplotlib.figure import Figure

from probatus.model.shap_dependence_plotter import DependencePlotter
from probatus.core import BaseFitComputePlotClass
from probatus.utils import (
    assure_list_of_strings,
    calculate_shap_importance,
    preprocess_data,
    preprocess_labels,
    get_single_scorer,
    shap_calc,
    Scorer,
    is_regression_model,
)


class ShapModelInterpreter(BaseFitComputePlotClass):
    """
    A wrapper class that allows easy analysis and interpretation of a model's features using SHAP values.

    This class provides methods to calculate and visualize SHAP (SHapley Additive exPlanations) values
    for machine learning models, helping to understand feature importance and interactions.

    Attributes:
        model (Any): The trained model to be interpreted
        scorer (Scorer): Scorer object used to evaluate model performance
        verbose (int): Controls verbosity of output
        random_state (Optional[int]): Random state for reproducibility
        fitted (bool): Indicates if the interpreter has been fitted
        X_train (pd.DataFrame): Training feature data
        X_test (pd.DataFrame): Test feature data
        y_train (pd.Series): Training target labels
        y_test (pd.Series): Test target labels
        column_names (List[str]): Feature names
        class_names (List[str]): Class names for classification
        train_score (float): Model score on training data
        test_score (float): Model score on test data
        shap_values_train (np.ndarray): SHAP values for training data
        shap_values_test (np.ndarray): SHAP values for test data
        expected_value_train (float): Expected value for training data
        expected_value_test (float): Expected value for test data
        tdp_train (DependencePlotter): Dependence plotter for training data
        tdp_test (DependencePlotter): Dependence plotter for test data
        importance_df (pd.DataFrame): DataFrame with feature importance metrics
        is_regression (bool): Whether the model is a regression model

    Example:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from probatus.interpret import ShapModelInterpreter
    import numpy as np
    import pandas as pd

    feature_names = ['f1', 'f2', 'f3', 'f4']

    # Prepare two samples
    X, y = make_classification(n_samples=5000, n_features=4, random_state=0)
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare and fit model. Remember about class_weight="balanced" or an equivalent.
    model = RandomForestClassifier(class_weight='balanced', n_estimators = 100, max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    # Train ShapModelInterpreter
    shap_interpreter = ShapModelInterpreter(model)
    feature_importance = shap_interpreter.fit_compute(X_train, X_test, y_train, y_test)

    # Make plots
    fig1 = shap_interpreter.plot('importance')
    fig2 = shap_interpreter.plot('summary')
    fig3 = shap_interpreter.plot('dependence', target_columns=['f1', 'f2'])
    fig4 = shap_interpreter.plot('sample', samples_index=[X_test.index.tolist()[0]])
    ```

    <img src="../img/model_interpret_importance.png" width="320" />
    <img src="../img/model_interpret_summary.png" width="320" />
    <img src="../img/model_interpret_dep.png" width="320" />
    <img src="../img/model_interpret_sample.png" width="320" />
    """

    def __init__(
        self,
        model: Any,
        scoring: Union[str, Scorer] = "roc_auc",
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize a ShapModelInterpreter object.

        Args:
            model (Any):
                The trained model to be interpreted. Must implement either predict or predict_proba
                method depending on the scoring metric requirements.

            scoring (Union[str, Scorer], default="roc_auc"):
                Metric for which the model performance is calculated. It can be either:
                - A string metric name aligned with predefined classification scorers in scikit-learn
                  (see: https://scikit-learn.org/stable/modules/model_evaluation.html)
                - An instance of probatus.utils.Scorer to define a custom metric

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
                - Default is `0`.

            random_state (Optional[int], default=None):
                Random state for reproducibility. If None, results will not be reproducible.
                For reproducible results, set it to an integer.
        """
        self.model = model
        self.scorer = get_single_scorer(scoring)
        self.verbose = verbose
        self.random_state = random_state
        self.fitted = False
        self.is_regression = False  # Will be set during fit

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **shap_kwargs: Any,
    ) -> "ShapModelInterpreter":
        """
        Fit the interpreter and calculate SHAP values for the provided datasets.

        This method preprocesses the input data, calculates model performance metrics,
        and computes SHAP values for both training and test datasets.

        Args:
            X_train (pd.DataFrame):
                DataFrame containing training feature data, of shape (n_samples, n_features).

            X_test (pd.DataFrame):
                DataFrame containing test feature data, of shape (n_samples, n_features).

            y_train (pd.Series):
                Series of target labels for training data, of shape (n_samples,).

            y_test (pd.Series):
                Series of target labels for test data, of shape (n_samples,).

            column_names (Optional[List[str]], default=None):
                List of feature names for the dataset. If None, column names from
                the X_train DataFrame are used.

            class_names (Optional[List[str]], default=None):
                List of class names e.g. ['neg', 'pos']. If None, the default
                ['Negative Class', 'Positive Class'] are used. For regression models,
                this parameter is ignored.

            **shap_kwargs:
                Keyword arguments passed to shap.Explainer. Notable parameters include:
                - approximate (bool): If True, uses faster but less accurate SHAP calculation
                - check_additivity (bool): If False, disables the additivity check inside SHAP
                For full details, see: https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html

        Returns:
            ShapModelInterpreter: The fitted instance (self).

        Raises:
            ValueError: If input data cannot be properly preprocessed
        """
        # Preprocess input data and ensure consistent format
        self.X_train, self.column_names = preprocess_data(
            X_train, X_name="X_train", column_names=column_names, verbose=self.verbose
        )
        self.X_test, _ = preprocess_data(X_test, X_name="X_test", column_names=column_names, verbose=self.verbose)
        self.y_train = preprocess_labels(y_train, index=self.X_train.index)
        self.y_test = preprocess_labels(y_test, index=self.X_test.index)

        # Determine if this is a regression model using the utility function
        self.is_regression = is_regression_model(self.model)

        # Set class names with default if not provided
        self.class_names = class_names
        if self.class_names is None:
            if self.is_regression:
                self.class_names = ["Regression Output"]
            else:
                self.class_names = ["Negative Class", "Positive Class"]

        # Calculate model performance metrics
        self.train_score = self.scorer.score(self.model, self.X_train, self.y_train)
        self.test_score = self.scorer.score(self.model, self.X_test, self.y_test)

        # Format results text for display in plots
        self.results_text = (
            f"Train {self.scorer.metric_name}: {np.round(self.train_score, 3)},\n"
            f"Test {self.scorer.metric_name}: {np.round(self.test_score, 3)}."
        )

        # Calculate SHAP values and related variables for training data
        (
            self.shap_values_train,
            self.expected_value_train,
            self.tdp_train,
        ) = self._prep_shap_related_variables(
            model=self.model,
            X=self.X_train,
            y=self.y_train,
            column_names=self.column_names,
            class_names=self.class_names,
            verbose=self.verbose,
            random_state=self.random_state,
            **shap_kwargs,
        )

        # Calculate SHAP values and related variables for test data
        (
            self.shap_values_test,
            self.expected_value_test,
            self.tdp_test,
        ) = self._prep_shap_related_variables(
            model=self.model,
            X=self.X_test,
            y=self.y_test,
            column_names=self.column_names,
            class_names=self.class_names,
            verbose=self.verbose,
            random_state=self.random_state,
            **shap_kwargs,
        )

        self.fitted = True

        # Return self for method chaining
        return self

    @staticmethod
    def _prep_shap_related_variables(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        approximate: bool = False,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **shap_kwargs: Any,
    ) -> Tuple[np.ndarray, float, DependencePlotter]:
        """
        Prepare SHAP-related variables used for model interpretation.

        This helper method calculates SHAP values, extracts the expected value,
        and initializes a DependencePlotter for a given dataset.

        Args:
            model (Any):
                The trained model to interpret. Must implement either predict or predict_proba
                method depending on the analysis requirements.

            X (pd.DataFrame):
                Feature data, of shape (n_samples, n_features).

            y (pd.Series):
                Target labels, of shape (n_samples,).

            approximate (bool, default=False):
                If True, uses faster but less accurate SHAP calculation.

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
                - Default is `0`.

            random_state (Optional[int], default=None):
                Random state for reproducibility.

            column_names (Optional[List[str]], default=None):
                List of feature names. If None, column names from X are used.

            class_names (Optional[List[str]], default=None):
                List of class names. If None, default class names are used.

            **shap_kwargs:
                Additional keyword arguments passed to shap.Explainer.

        Returns:
            Tuple[np.ndarray, float, DependencePlotter]:
                - SHAP values array of shape (n_samples, n_features)
                - Expected value of the explainer
                - Fitted DependencePlotter instance
        """
        # Calculate SHAP values and get the explainer
        shap_values, explainer = shap_calc(
            model,
            X,
            approximate=approximate,
            verbose=verbose,
            random_state=random_state,
            return_explainer=True,
            **shap_kwargs,
        )

        expected_value = explainer.expected_value

        # For sklearn models, the expected value consists of n elements
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]

        # Initialize and fit the dependence plotter for visualizing feature interactions
        tdp = DependencePlotter(model, verbose=verbose).fit(
            X,
            y,
            column_names=column_names,
            class_names=class_names,
            precalc_shap=shap_values,
        )
        return shap_values, expected_value, tdp

    def compute(
        self, return_scores: bool = False, shap_variance_penalty_factor: Optional[float] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
        """
        Compute feature importance based on SHAP values.

        This method calculates and aggregates SHAP values to create a DataFrame
        showing the importance of each feature in the model.

        Args:
            return_scores (bool, default=False):
                If True, returns the train and test scores along with the feature importance DataFrame.
                If False, returns only the feature importance DataFrame.

            shap_variance_penalty_factor (Optional[float], default=None):
                Factor to penalize features with high variance in SHAP values.
                This promotes features with more consistent impact across samples.
                Recommended values are between 0.5 and 1.0.
                Formula: penalized_shap_mean = (mean_shap - (std_shap * shap_variance_penalty_factor))

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
                - If return_scores=False: DataFrame with SHAP feature importance metrics
                - If return_scores=True: Tuple containing (importance_df, train_score, test_score)

        Raises:
            ValueError: If the interpreter has not been fitted
        """
        self._check_if_fitted()

        # Compute SHAP importance
        self.importance_df_train = calculate_shap_importance(
            self.shap_values_train,
            self.column_names,
            output_columns_suffix="_train",
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        self.importance_df_test = calculate_shap_importance(
            self.shap_values_test,
            self.column_names,
            output_columns_suffix="_test",
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        # Combine train and test results, sort by test importance, and select relevant columns
        self.importance_df = pd.concat([self.importance_df_train, self.importance_df_test], axis=1).sort_values(
            "mean_abs_shap_value_test", ascending=False
        )[
            [
                "mean_abs_shap_value_test",
                "mean_abs_shap_value_train",
                "mean_shap_value_test",
                "mean_shap_value_train",
            ]
        ]

        if return_scores:
            return self.importance_df, self.train_score, self.test_score
        else:
            return self.importance_df

    def fit_compute(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        return_scores: bool = False,
        shap_variance_penalty_factor: Optional[float] = None,
        **shap_kwargs: Any,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
        """
        Fit the interpreter and compute feature importance in a single step.

        This convenience method combines the fit() and compute() methods to streamline
        the workflow for model interpretation.

        Args:
            X_train (pd.DataFrame):
                DataFrame containing training feature data, of shape (n_samples, n_features).

            X_test (pd.DataFrame):
                DataFrame containing test feature data, of shape (n_samples, n_features).

            y_train (pd.Series):
                Series of target labels for training data, of shape (n_samples,).

            y_test (pd.Series):
                Series of target labels for test data, of shape (n_samples,).

            column_names (Optional[List[str]], default=None):
                List of feature names for the dataset. If None, column names from
                the X_train DataFrame are used.

            class_names (Optional[List[str]], default=None):
                List of class names e.g. ['neg', 'pos']. If None, the default
                ['Negative Class', 'Positive Class'] are used.

            return_scores (bool, default=False):
                If True, returns the train and test scores along with the feature importance DataFrame.
                If False, returns only the feature importance DataFrame.

            shap_variance_penalty_factor (Optional[float], default=None):
                Factor to penalize features with high variance in SHAP values.
                This promotes features with more consistent impact across samples.
                Recommended values are between 0.5 and 1.0.
                Formula: penalized_shap_mean = (mean_shap - (std_shap * shap_variance_penalty_factor))

            **shap_kwargs:
                Keyword arguments passed to shap.Explainer. Notable parameters include:
                - approximate (bool): If True, uses faster but less accurate SHAP calculation
                - check_additivity (bool): If False, disables the additivity check inside SHAP
                For full details, see: https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
                - If return_scores=False: DataFrame with SHAP feature importance metrics
                - If return_scores=True: Tuple containing (importance_df, train_score, test_score)

        Raises:
            ValueError: If input data cannot be properly preprocessed
        """
        self.fit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            column_names=column_names,
            class_names=class_names,
            **shap_kwargs,
        )
        return self.compute(return_scores=return_scores, shap_variance_penalty_factor=shap_variance_penalty_factor)

    def plot(
        self,
        plot_type: Literal["importance", "summary", "dependence", "sample"],
        target_set: Literal["train", "test"] = "test",
        target_columns: Optional[Union[str, List[str]]] = None,
        samples_index: Optional[Union[int, str, List, pd.Index]] = None,
        show: bool = False,
        **plot_kwargs: Any,
    ) -> Union[Figure, List[Figure]]:
        """
        Generate SHAP-based visualizations for model interpretation.

        This method creates various types of plots to help understand feature importance,
        feature interactions, and individual predictions.

        Args:
            plot_type (Literal["importance", "summary", "dependence", "sample"]):
                Type of plot to generate. Must be one of:
                - 'importance': Bar plot showing feature importance
                - 'summary': Dot plot showing feature impact distribution
                - 'dependence': Plots showing feature interactions
                - 'sample': Waterfall plot explaining individual predictions

            target_set (Literal["train", "test"], default="test"):
                Dataset to use for plotting, either "train" or "test".
                Using the test set is recommended to avoid training data bias.

            target_columns (Optional[Union[str, List[str]]], default=None):
                Features to include in the plot. If None, all features are used.
                For 'dependence' plots, this specifies which features to create plots for.

            samples_index (Optional[Union[int, str, List, pd.Index]], default=None):
                Sample indices to explain when plot_type='sample'.
                Required for 'sample' plots, ignored for other plot types.

            show (bool, default=False):
                If True, displays the plot immediately.
                If False, returns the plot figure without showing, allowing for further customization.

            **plot_kwargs:
                Additional keyword arguments passed to the underlying plotting functions:
                - For 'importance' and 'summary': passed to shap.summary_plot()
                - For 'dependence': passed to DependencePlotter.plot()
                - For 'sample': passed to shap.plots.waterfall()

        Returns:
            Union[Figure, List[Figure]]:
                Matplotlib Figure object(s) containing the generated plot(s).
                Returns a single Figure for 'importance' and 'summary' plots.
                Returns a list of Figures for 'dependence' and 'sample' plots with multiple features/samples.

        Raises:
            ValueError: If samples_index is not provided for 'sample' plots
            TypeError: If samples_index has an invalid type for 'sample' plots
        """
        self._check_if_fitted()

        # Prepare data and select appropriate dataset
        target_columns = self._prepare_target_columns(target_columns)
        target_columns_indices = [self.column_names.index(col) for col in target_columns]

        # Select dataset based on target_set parameter
        target_data = self._select_target_dataset(target_set)
        target_X, target_shap_values = target_data["X"], target_data["shap_values"]
        target_tdp, target_expected_value = target_data["tdp"], target_data["expected_value"]

        # Generate the appropriate plot based on plot_type
        if plot_type in ["importance", "summary"]:
            return self._create_summary_plot(
                plot_type,  # type: ignore[arg-type]
                target_set,
                target_X,
                target_shap_values,
                target_columns,
                target_columns_indices,
                show,
                **plot_kwargs,
            )
        elif plot_type == "dependence":
            return self._create_dependence_plots(target_columns, target_tdp, show, **plot_kwargs)
        elif plot_type == "sample":
            return self._create_sample_plots(
                samples_index,
                target_X,
                target_shap_values,
                target_expected_value,
                target_columns,
                target_set,
                show,
                **plot_kwargs,
            )
        else:
            raise ValueError("Wrong plot type, select from 'importance', 'summary', 'dependence', or 'sample'")

    def _prepare_target_columns(self, target_columns: Optional[Union[str, List[str]]]) -> List[str]:
        """
        Prepare target columns list for plotting.

        This helper method ensures that target_columns is a list of strings.
        If None is provided, all column names are used.

        Args:
            target_columns (Optional[Union[str, List[str]]]):
                Column names to include. Can be a single string, a list of strings, or None.

        Returns:
            List[str]: List of column names to use in plots
        """
        if target_columns is None:
            target_columns = self.column_names
        return assure_list_of_strings(target_columns, "target_columns")

    def _select_target_dataset(self, target_set: Literal["train", "test"]) -> dict:
        """
        Select the appropriate dataset based on target_set parameter.

        This helper method returns the data, SHAP values, and related objects
        for either the training or test dataset.

        Args:
            target_set (Literal["train", "test"]):
                Which dataset to use, either "train" or "test".

        Returns:
            dict: Dictionary containing the selected dataset components:
                - "X": Feature data (pd.DataFrame)
                - "shap_values": SHAP values (np.ndarray)
                - "tdp": DependencePlotter instance
                - "expected_value": Expected value for SHAP calculations

        Raises:
            ValueError: If target_set is not "train" or "test"
        """
        if target_set == "test":
            return {
                "X": self.X_test,
                "shap_values": self.shap_values_test,
                "tdp": self.tdp_test,
                "expected_value": self.expected_value_test,
            }
        elif target_set == "train":
            return {
                "X": self.X_train,
                "shap_values": self.shap_values_train,
                "tdp": self.tdp_train,
                "expected_value": self.expected_value_train,
            }
        else:
            raise ValueError('The target_set parameter can be either "train" or "test".')

    def _create_summary_plot(
        self,
        plot_type: Literal["importance", "summary"],
        target_set: Literal["train", "test"],
        target_X: pd.DataFrame,
        target_shap_values: pd.DataFrame,
        target_columns: List[str],
        target_columns_indices: List[int],
        show: bool,
        **plot_kwargs: Any,
    ) -> Figure:
        """
        Create importance or summary plots based on SHAP values.

        This helper method generates bar plots (for importance) or dot plots (for summary)
        to visualize feature importance and impact distributions.

        Args:
            plot_type (Literal["importance", "summary"]):
                Type of plot to create, either "importance" or "summary".

            target_set (Literal["train", "test"]):
                Dataset being used, either "train" or "test".

            target_X (pd.DataFrame):
                Feature data for the selected dataset.

            target_shap_values (pd.DataFrame):
                SHAP values for the selected dataset.

            target_columns (List[str]):
                List of column names to include in the plot.

            target_columns_indices (List[int]):
                Indices of the target columns in the original data.

            show (bool):
                Whether to display the plot immediately.

            **plot_kwargs:
                Additional keyword arguments passed to shap.summary_plot().

        Returns:
            Figure: Matplotlib Figure object containing the generated plot
        """
        # Filter data to include only the target columns
        target_X = target_X[target_columns]

        # Handle different types of target_shap_values
        if isinstance(target_shap_values, pd.DataFrame):
            # If it's a DataFrame, select columns by name
            target_shap_values = target_shap_values[target_columns].values
        else:
            # If it's a numpy array, select columns by index
            target_shap_values = target_shap_values[:, target_columns_indices]

        # Configure plot type and title
        plot_style = "bar" if plot_type == "importance" else "dot"
        model_type = "Regression" if self.is_regression else "Feature Importance"
        plot_title = f"SHAP {model_type if plot_type == 'importance' else 'Summary plot'} for {target_set} set"

        # Create the plot - for regression models, don't pass class_names
        if self.is_regression:
            summary_plot(
                target_shap_values,
                target_X,
                plot_type=plot_style,
                show=False,
                **plot_kwargs,
            )
        else:
            summary_plot(
                target_shap_values,
                target_X,
                plot_type=plot_style,
                class_names=self.class_names,
                show=False,
                **plot_kwargs,
            )

        # Get the current figure and adjust layout to make room for title
        fig = plt.gcf()

        # Customize the plot
        ax = plt.gca()
        ax.set_title(plot_title, pad=20)  # Add padding to the title

        # Add model performance metrics as annotation
        ax.annotate(
            self.results_text,
            (0, 0),
            (0, -50),
            fontsize=12,
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

        # Apply layout adjustments after all elements are added
        # This ensures proper spacing for all elements including title and annotations
        fig.tight_layout()
        # Add extra padding at the top for the title
        fig.subplots_adjust(top=0.85)

        if show:
            plt.show()

        return fig

    def _create_dependence_plots(
        self, target_columns: List[str], target_tdp: DependencePlotter, show: bool, **plot_kwargs: Any
    ) -> Union[Figure, List[Figure]]:
        """
        Create dependence plots to visualize feature interactions.

        This helper method generates plots showing how SHAP values depend on feature values,
        helping to understand feature interactions and their impact on predictions.

        Args:
            target_columns (List[str]):
                List of features to create dependence plots for.

            target_tdp (DependencePlotter):
                Fitted DependencePlotter instance for the selected dataset.

            show (bool):
                Whether to display the plots immediately.

            **plot_kwargs:
                Additional keyword arguments passed to DependencePlotter.plot().

        Returns:
            Union[Figure, List[Figure]]:
                Matplotlib Figure object(s) containing the generated plot(s).
                Returns a single Figure if there's only one plot, otherwise a list of Figures.
        """
        figures_list: List[Figure] = []
        for feature_name in target_columns:
            # The plot method now returns a list of Figure objects
            fig_result = target_tdp.plot(feature=feature_name, figsize=(10, 7), show=False, **plot_kwargs)

            # Add the figures to our list
            if isinstance(fig_result, list):
                figures_list.extend(fig_result)
            else:
                figures_list.append(fig_result)

        # Return a single Figure if there's only one plot
        if len(figures_list) == 1:
            return figures_list[0]
        return figures_list

    def _create_sample_plots(
        self,
        samples_index: Optional[Union[int, str, List, pd.Index]],
        target_X: pd.DataFrame,
        target_shap_values: pd.DataFrame,
        target_expected_value: float,
        target_columns: List[str],
        target_set: Literal["train", "test"],
        show: bool,
        **plot_kwargs: Any,
    ) -> Union[Figure, List[Figure]]:
        """
        Create waterfall plots explaining individual predictions.

        This helper method generates plots showing how each feature contributes
        to the prediction for specific samples in the dataset.

        Args:
            samples_index (Optional[Union[int, str, List, pd.Index]]):
                Indices of samples to explain.

            target_X (pd.DataFrame):
                Feature data for the selected dataset.

            target_shap_values (pd.DataFrame):
                SHAP values for the selected dataset.

            target_expected_value (float):
                Expected value (base value) for SHAP calculations.

            target_columns (List[str]):
                List of column names to include in the plot.

            target_set (Literal["train", "test"]):
                Dataset being used, either "train" or "test".

            show (bool):
                Whether to display the plots immediately.

            **plot_kwargs:
                Additional keyword arguments passed to shap.plots.waterfall().

        Returns:
            Union[Figure, List[Figure]]:
                Matplotlib Figure object(s) containing the generated plot(s).
                Returns a single Figure if there's only one plot, otherwise a list of Figures.

        Raises:
            ValueError: If samples_index is None
        """
        # Validate samples_index parameter
        if samples_index is None:
            raise ValueError("For sample plot, you need to specify the samples_index to be plotted")

        # Convert scalar indices to list for consistent handling
        if not isinstance(samples_index, (list, pd.Index)):
            samples_index = [samples_index]

        figures_list: List[Figure] = []

        # Create a waterfall plot for each sample
        for sample_index in samples_index:
            # Get the position of the sample in the DataFrame
            sample_loc = target_X.index.get_loc(sample_index)

            # Calculate appropriate figure dimensions
            max_name_length = max([len(str(name)) for name in target_columns])
            fig_width = max(10, 8 + max_name_length * 0.1)

            # Get SHAP values for this sample
            if isinstance(target_shap_values, pd.DataFrame):
                # If it's a DataFrame, get the row by index
                sample_shap_values = target_shap_values.loc[sample_index].values
            else:
                # If it's a numpy array, get the row by position
                sample_shap_values = target_shap_values[sample_loc, :]

            # Create a SHAP Explanation object
            explanation = Explanation(
                values=sample_shap_values,
                base_values=target_expected_value,
                data=target_X.loc[sample_index].values,
                feature_names=target_columns,
            )

            # Extract max_display from plot_kwargs
            max_display = plot_kwargs.pop("max_display", 10) if "max_display" in plot_kwargs else 10

            # Close any existing figures and create the waterfall plot
            plt.close("all")
            waterfall(
                explanation,
                show=False,
                max_display=min(len(target_columns), max_display),
                **plot_kwargs,
            )

            # Customize the plot
            fig = plt.gcf()
            current_ax = plt.gca()
            fig.set_size_inches(fig_width, 8)

            model_type = "Regression" if self.is_regression else "Sample Explanation"
            plot_title = f"SHAP {model_type} of {target_set} sample for index={sample_index}"
            current_ax.set_title(plot_title, pad=20)  # Add padding to the title
            current_ax.tick_params(axis="y", labelsize=10)

            # Apply consistent layout adjustments
            # First adjust left margin for feature names
            plt.subplots_adjust(left=max(0.2, 0.15 + max_name_length * 0.01))
            # Then apply tight layout for overall spacing
            plt.tight_layout()
            # Finally add space at the top for the title
            plt.subplots_adjust(top=0.9)

            figures_list.append(fig)

            if show:
                plt.show()

        # Return a single Figure if there's only one plot
        if len(figures_list) == 1:
            return figures_list[0]
        return figures_list
