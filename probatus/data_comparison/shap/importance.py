from probatus.data_comparison._base import BaseResemblanceModel
from probatus._wrapper import (
    calculate_shap_explanation,
    calculate_shap_importance_dataframe,
    extract_multiclass_shap_parameters,
    shap_explanation_to_shap_values,
)
from probatus.metrics import Scorer


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from shap import Explanation
from shap.plots import bar, beeswarm, waterfall
from sklearn.base import BaseEstimator


from typing import Any, List, Literal, Optional, Union


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
        multi_class_kwargs, shap_kwargs = extract_multiclass_shap_parameters(shap_kwargs)

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
        self.shap_values_test = shap_explanation_to_shap_values(
            shap_explanation=self.shap_explanation_test,
            model=self.model,
            X=self.X_test,
            **multi_class_kwargs,
        )

        # Calculate feature importance from SHAP values
        shap_df = pd.DataFrame(self.shap_values_test, columns=self.column_names)
        self.report_df = calculate_shap_importance_dataframe(shap_df, self.column_names)

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
