import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from typing import Any, List, Optional, Tuple, Union, Literal, Dict, cast
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shap import Explanation

from probatus.core import BaseFitComputePlotClass
from probatus.utils import (
    preprocess_data,
    preprocess_labels,
    calculate_shap_explanation,
    is_regression_model,
    handle_class_names,
    extract_shap_multiclass_params,
    shap_explanation_to_shap_df,
    is_multiclass_model,
)
from probatus.utils.common import get_pipeline_estimator_and_preprocessor


class DependencePlotter(BaseFitComputePlotClass):
    """
    Plotter used to plot SHAP dependence plot together with the target rates.

    This class creates visualizations that show how SHAP values for a specific feature
    relate to the feature's values, along with the target rate distribution. It helps
    understand how a feature influences model predictions across different feature values.

    Currently it supports tree-based and linear models.

    Args:
        model (Any):
            A fitted classifier model for which interpretation is done. Must implement
            predict_proba or decision_function methods depending on the model type.

        verbose (int, optional):
            Controls verbosity of the output (0-2). Default is 0.
            - 0: No output or warnings
            - 1: Only important warnings
            - 2: All prints and warnings

        random_state (Optional[int], optional):
            Random state for reproducibility. If None, results may not be reproducible.
            Default is None.

    Raises:
        ValueError:
            If invalid parameters are provided during plotting.

        RuntimeError:
            If methods requiring a fitted model are called before fitting.

    Example:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    from probatus.model import DependencePlotter

    # Create sample data with named features
    X, y = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    feature_names = ['feature_1', 'feature_2', 'feature_3']
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y)

    # Train a model
    model = RandomForestClassifier(random_state=42).fit(X, y)

    # Initialize and fit the plotter
    dep_plotter = DependencePlotter(model)
    shap_values = dep_plotter.fit_compute(X, y, column_names=feature_names)

    # Plot the dependence for a specific feature
    dep_plotter.plot(feature='feature_3')
    ```

    <img src="../img/model_interpret_dep.png"/>
    """

    def __init__(
        self, model: Union[BaseEstimator, Pipeline], verbose: Literal[0, 1, 2] = 0, random_state: Optional[int] = None
    ) -> None:
        """
        Initializes the DependencePlotter class.

        Args:
            model (Union[BaseEstimator, Pipeline]):
                A fitted regression or classification model or pipeline. Must implement
                predict_proba or decision_function methods depending on the model type.

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
                - Default is `0`.

            random_state (Optional[int], optional):
                Random state for reproducibility. If None, results may not be reproducible.
                Default is None.
        """
        self.model, self.preprocessor = get_pipeline_estimator_and_preprocessor(model)
        self.verbose: Literal[0, 1, 2] = verbose
        self.random_state: Optional[int] = random_state
        self.fitted: bool = False
        self.class_names: List[str] = None
        self.is_regression: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        column_names: Optional[List[str]] = None,
        class_names: Optional[Union[List[str], Dict[Union[int, str], str]]] = None,
        precalc_shap_explanation: Optional[Explanation] = None,
        **shap_kwargs: Any,
    ) -> "DependencePlotter":
        """
        Fits the plotter to the model and data by computing the SHAP values.

        This method preprocesses the input data and computes SHAP values for the model.
        If precalculated SHAP values are provided, they are used directly.

        Args:
            X (pd.DataFrame):
                Input feature dataset of shape (n_samples, n_features).
                Must be a pandas DataFrame.

            y (pd.Series):
                Target variable of shape (n_samples,).
                Must be a pandas Series with binary values (0, 1).

            column_names (Optional[List[str]], optional):
                List of feature names for the dataset. If None, column names from
                the X dataframe are used. Default is None.

            class_names (Optional[Union[List[str], Dict[Union[int, str], str]]], optional):
                Either a list of class names (e.g. ['blue', 'red', 'green']) that will be mapped
                to the sorted unique values in y, or a dictionary mapping target values to class
                names (e.g. {0: 'blue', 1: 'red', 2: 'green'}).
                If None, default labels will be 'label_0', 'label_1', etc. for classification
                or 'Regression Output' for regression. Default is None.

            precalc_shap_explanation (Optional[Explanation], optional):
                Precalculated SHAP explanation object.
                If provided, it is used directly instead of computing new ones.
                Default is None.

            **shap_kwargs (Any):
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values multi-classification conversion - parameters like 'class_selection', 'multi-class_aggregation', and 'weights'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multi-class models.

        Returns:
            DependencePlotter: The fitted instance with computed SHAP values.

        Raises:
            ValueError: If input data formats are invalid.
        """
        # Transform data if model is a Pipeline
        if self.preprocessor is not None:
            column_names = X.columns if column_names is None else column_names
            X = self.preprocessor.transform(X)

        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, index=self.X.index)

        # Determine if this is a regression model
        self.is_regression = is_regression_model(self.model)
        self.is_multiclass = is_multiclass_model(self.model, self.y)

        # Use class names for plotting
        self.class_names = handle_class_names(self.y, class_names, self.is_regression)

        # Split arguments for multi-classification
        multi_class_kwargs, shap_kwargs = extract_shap_multiclass_params(shap_kwargs)

        # Calculate SHAP values
        if precalc_shap_explanation is not None:
            # Use precalculated SHAP values
            self.shap_explanation = precalc_shap_explanation
        else:
            # Calculate SHAP values
            self.shap_explanation = calculate_shap_explanation(
                self.model,
                self.X,
                return_explainer=False,
                verbose=self.verbose,
                random_state=self.random_state,
                **shap_kwargs,
            )

        # Set the SHAP values param
        self.shap_values = shap_explanation_to_shap_df(
            shap_explanation=self.shap_explanation,
            model=self.model,
            X=self.X,
            **multi_class_kwargs,
        )

        # Set default values for quantile range and alpha
        self.min_q: float = 0.0
        self.max_q: float = 1.0
        self.alpha: float = 1.0

        self.fitted = True
        return self

    def compute(self) -> pd.DataFrame:
        """
        Computes the report returned to the user, namely the SHAP values generated on the dataset.

        This method returns the previously computed SHAP values as a DataFrame.
        It must be called after the model has been fitted.

        Returns:
            pd.DataFrame: DataFrame containing SHAP values for each feature in X.
                          Each column corresponds to a feature, and each row to an observation.

        Raises:
            RuntimeError: If called before the model is fitted.
        """
        self._check_if_fitted()
        return self.shap_values

    def fit_compute(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        column_names: Optional[List[str]] = None,
        class_names: Optional[Union[List[str], Dict[Union[int, str], str]]] = None,
        precalc_shap: Optional[pd.DataFrame] = None,
        **shap_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fits the plotter to the model and data, then returns the computed SHAP values.

        This is a convenience method that combines fit() and compute() in one call.

        Args:
            X (pd.DataFrame):
                Input feature dataset of shape (n_samples, n_features).
                Must be a pandas DataFrame.

            y (pd.Series):
                Target variable of shape (n_samples,).
                Must be a pandas Series with binary values (0, 1).

            column_names (Optional[List[str]], optional):
                List of feature names for the dataset. If None, column names from
                the X dataframe are used. Default is None.

            class_names (Optional[Union[List[str], Dict[Union[int, str], str]]], optional):
                Either a list of class names (e.g. ['blue', 'red', 'green']) that will be mapped
                to the sorted unique values in y, or a dictionary mapping target values to class
                names (e.g. {0: 'blue', 1: 'red', 2: 'green'}).
                If None, default labels will be 'label_0', 'label_1', etc. for classification
                or 'Regression Output' for regression. Default is None.

            precalc_shap (Optional[pd.DataFrame], optional):
                Precalculated SHAP values of shape (n_samples, n_features).
                If provided, they are used directly instead of computing new ones.
                Default is None.

            **shap_kwargs (Any):
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values conversion - parameters like 'class_selection', 'multi-class_aggregation', and 'weights'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multi-class models.

        Returns:
            pd.DataFrame: DataFrame containing SHAP values for each feature in X.
                          Each column corresponds to a feature, and each row to an observation.

        Raises:
            ValueError: If input data formats are invalid.
            RuntimeError: If internal computation fails.
        """
        self.fit(
            X,
            y,
            column_names=column_names,
            class_names=class_names,
            precalc_shap_explanation=precalc_shap,
            **shap_kwargs,
        )
        return self.compute()

    def plot(
        self,
        feature: Union[str, int],
        figsize: Tuple[float, float] = (15, 10),
        bins: Union[int, List[float]] = 10,
        show: bool = False,
        min_q: float = 0,
        max_q: float = 1,
        alpha: float = 1.0,
    ) -> Figure:
        """
        Plots the SHAP values for data points for a given feature, along with target rate and value distribution.

        Creates a two-panel plot:
        - Top panel: SHAP dependence plot showing how SHAP values relate to feature values
        - Bottom panel: Feature value distribution and target rate

        Args:
            feature (Union[str, int]):
                Feature name or index of the feature to be analyzed. If an integer is provided,
                it is used as an index into the column_names list.

            figsize (Tuple[float, float], optional):
                Tuple specifying size (width, height) of resulting figure in inches.
                Default is (15, 10).

            bins (Union[int, List[float]], optional):
                Number of bins or boundaries of bins (supplied in list) for target-rate plot.
                Default is 10.

            show (bool, optional):
                If True, the plots are shown to the user, otherwise they are not shown.
                Not showing plot can be useful when you want to edit the returned axes before showing them.
                Default is False.

            min_q (float, optional):
                Minimum quantile from which to consider values, used for filtering outliers.
                Must be between 0 and 1, and less than max_q. Default is 0.

            max_q (float, optional):
                Maximum quantile until which data points are considered, used for filtering outliers.
                Must be between 0 and 1, and greater than min_q. Default is 1.

            alpha (float, optional):
                Alpha blending value for scatter points, between 0 (transparent) and 1 (opaque).
                Default is 1.0.

        Returns:
            matplotlib.figure.Figure: Figure containing both the SHAP dependence plot and target rate plot.

        Raises:
            ValueError:
                - If min_q >= max_q
                - If feature is not recognized in the dataset
                - If alpha is not between 0 and 1
                - If feature index is out of range

            RuntimeError:
                - If called before the model is fitted
        """
        self._check_if_fitted()

        # Validate input parameters
        if min_q >= max_q:
            raise ValueError("min_q must be smaller than max_q")
        if isinstance(feature, (int, np.int64)):
            if feature >= len(self.column_names):
                raise ValueError(f"Feature index {feature} out of range (0-{len(self.column_names) - 1})")
            feature_name = self.column_names[feature]
        else:
            feature_name = feature
            if feature_name not in self.X.columns:
                raise ValueError(f"Feature '{feature_name}' not recognized in the dataset")
        if (alpha < 0) or (alpha > 1):
            raise ValueError("alpha must be a float value between 0 and 1")

        self.min_q, self.max_q, self.alpha = min_q, max_q, alpha

        # Create figure with two subplots (2:1 ratio)
        fig: Figure = plt.figure(figsize=figsize)
        ax1: Axes = plt.subplot2grid((3, 1), (0, 0), rowspan=2, fig=fig)
        ax2: Axes = plt.subplot2grid((3, 1), (2, 0), fig=fig)

        # Call the plotting methods with the created axes
        self._dependence_plot(feature=feature_name, ax=ax1)
        self._target_rate_plot(feature=feature_name, bins=bins, ax=ax2)

        # Ensure both plots have the same x-axis limits for visual consistency
        ax2.set_xlim(ax1.get_xlim())

        if show:
            plt.show()

        return fig

    def _dependence_plot(self, feature: str, ax: Optional[Axes] = None) -> Figure:
        """
        Plots SHAP values for data points with respect to the specified feature.

        Creates a scatter plot showing how SHAP values relate to feature values,
        with points colored by their target class.

        Args:
            feature (str):
                Feature name for which dependence plot is to be created.

            ax (Optional[matplotlib.axes.Axes], optional):
                Axis on which to draw plot. If None, current axis will be used.
                Default is None.

        Returns:
            matplotlib.figure.Figure: Figure containing the dependence plot.

        Raises:
            ValueError: If feature is not found in the dataset.
            RuntimeError: If called before the model is fitted.
        """
        if isinstance(feature, (int, np.int64)):
            feature = self.column_names[int(feature)]

        # Get filtered data based on quantile range
        X, y, shap_val = self._get_X_y_shap_with_q_cut(feature=feature)

        # Create or use provided axes
        if ax is None:
            plt.style.use("default")  # Reset to default style first
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = cast(Figure, ax.figure)
            ax = cast(Axes, ax)

        # Set light gray background with white grid
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("white")

        # Add subtle grid lines
        ax.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

        if self.is_regression:
            # For regression, use a single scatter plot with a colormap based on target values
            scatter = ax.scatter(X, shap_val, c=y, cmap="viridis", alpha=self.alpha, label=self.class_names[0])
            # Add a colorbar to show the target value scale
            cbar = plt.colorbar(scatter, ax=ax, label="Target Value")
            cbar.outline.set_linewidth(0.8)
            cbar.outline.set_edgecolor("lightgray")
        else:
            # For classification, create separate scatter plot for each class
            # Sort the unique values of y and use the sorted values to map the class names
            # Define SHAP-like colors
            shap_colors = ["#1E88E5", "#ff0051"]  # Blue and red, the main SHAP colors

            for i, (class_name, class_value) in enumerate(zip(self.class_names, sorted(self.y.unique()))):
                if self.is_multiclass:
                    ax.scatter(
                        X[y == class_value],
                        shap_val[y == class_value][:, i],
                        label=class_name,
                        alpha=self.alpha,
                        color=shap_colors[i % len(shap_colors)],  # Cycle through colors if more than 2 classes
                    )
                else:
                    ax.scatter(
                        X[y == class_value],
                        shap_val[y == class_value],
                        label=class_name,
                        alpha=self.alpha,
                        color=shap_colors[i % len(shap_colors)],  # Cycle through colors if more than 2 classes
                    )

        # Set custom tick parameters
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Improve axis labels
        ax.set_ylabel("SHAP value", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{feature}", fontsize=11, fontweight="bold")
        ax.set_title(f"Dependence plot for {feature} feature", fontsize=13, fontweight="bold", pad=15)

        # Create a styled legend
        legend = ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="lightgray", fontsize=10)
        legend.get_frame().set_facecolor("white")

        # Add a thin border
        for spine in ax.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.8)

        return fig

    def _target_rate_plot(
        self, feature: Union[str, int], bins: Union[int, List[float]] = 10, ax: Optional[Axes] = None
    ) -> Tuple[Union[List[float], np.ndarray], Figure, pd.Series]:
        """
        Plots the distribution of the feature values and the target rate as function of the feature.

        This creates a histogram of feature values with a line showing the target rate
        (proportion of positive class) for each bin of feature values.

        For regression models, it shows the mean target value for each bin instead of a target rate.

        Args:
            feature (Union[str, int]):
                Feature name or index for which to create target rate plot.
                If an integer is provided, it is converted to the corresponding feature name.

            bins (Union[int, List[float]], optional):
                Number of bins or boundaries of desired bins in list.
                Default is 10.

            ax (Optional[matplotlib.axes.Axes], optional):
                Axis on which to draw plot. If None, current axis will be used.
                Default is None.

        Returns:
            Tuple[Union[List[float], np.ndarray], matplotlib.figure.Figure, pd.Series]:
                - Boundaries of bins used for the histogram
                - Figure containing the target rate plot
                - Series containing target ratio (proportion of positive class) for each bin
                  or mean target value for regression models

        Raises:
            ValueError: If feature is not found in the dataset.
            RuntimeError: If called before the model is fitted.
            TypeError: If bins is not an integer or a list of floats.
        """
        # Create or use provided axes
        if ax is None:
            plt.style.use("default")  # Reset to default style first
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = cast(Figure, ax.figure)
            ax = cast(Axes, ax)

        # Set light gray background with white grid
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("white")

        # Add subtle grid lines
        ax.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

        # Handle feature name extraction
        feature_name: str = (
            self.column_names[feature]
            if isinstance(feature, (int, np.int64)) and isinstance(self.column_names, list)
            else str(feature)
        )

        # Get filtered data based on quantile range
        x, y, _ = self._get_X_y_shap_with_q_cut(feature=feature_name)

        # Create bins if not explicitly supplied
        bin_edges: Union[List[float], np.ndarray]
        if isinstance(bins, (int, np.int64)):
            # Use KBinsDiscretizer to create uniform bins
            simple_binner = KBinsDiscretizer(n_bins=int(bins), encode="ordinal", strategy="uniform").fit(
                np.array(x).reshape(-1, 1)
            )
            bin_edges = simple_binner.bin_edges_[0]
            # Set first and last bin edges to infinity to ensure all data points are included
            if isinstance(bin_edges, np.ndarray):
                bin_edges = bin_edges.tolist()  # Convert to list for easier manipulation
            bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        elif isinstance(bins, list):
            bin_edges = bins
        else:
            # Handle invalid bin types (like float)
            raise TypeError(f"bins must be an integer or a list of floats, got {type(bins).__name__}")

        # Convert bin_edges to numpy array for easier manipulation
        bin_edges_array = np.array(bin_edges)

        # Determine bin for each datapoint
        # Add 1 to the last bin edge to ensure np.digitize includes the maximum value
        bin_edges_for_digitize = bin_edges_array.copy()
        bin_edges_for_digitize[-1] = bin_edges_for_digitize[-1] + 1
        indices = np.digitize(x, bin_edges_for_digitize)

        # Create dataframe with binned data and group by bin index
        dfs = pd.DataFrame({feature_name: x, "y": y, "bin_index": pd.Series(indices, index=x.index)}).groupby(
            "bin_index", as_index=True
        )

        # Extract target ratio (mean of y) and mean feature value for each bin
        target_ratio = dfs["y"].mean()  # Proportion of positive class in each bin or mean target value for regression
        x_vals = dfs[feature_name].mean()  # Mean feature value in each bin

        # Transform the first and last bin to work with plt.hist method
        # Replace infinity values with actual min/max for plotting
        bin_edges_for_hist = bin_edges_array.copy()
        if bin_edges_for_hist[0] == -np.inf:
            bin_edges_for_hist[0] = x.min()
        if bin_edges_for_hist[-1] == np.inf:
            bin_edges_for_hist[-1] = x.max()

        # Plot histogram of feature values with SHAP-like colors
        hist_color = "#1E88E5"  # SHAP blue color
        _, bins, _ = ax.hist(
            x, bins=cast(Union[int, List[float]], bin_edges_for_hist), lw=1, alpha=0.6, color=hist_color
        )
        ax.set_ylabel("Counts", fontsize=11, fontweight="bold")

        # Set custom tick parameters
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Create twin axis for target rate line
        ax2 = ax.twinx()
        ax2 = cast(Axes, ax2)

        if self.is_regression:
            # For regression, show mean target value
            line_color = "#ff0051"  # SHAP red color
            ax2.plot(x_vals, target_ratio, color=line_color, linewidth=2)
            ax2.set_ylabel("Mean target value", color=line_color, fontsize=11, fontweight="bold")
        else:
            # For classification, show target rate (proportion of positive class)
            line_color = "#ff0051"  # SHAP red color
            ax2.plot(x_vals, target_ratio, color=line_color, linewidth=2)
            ax2.set_ylabel("Target rate", color=line_color, fontsize=11, fontweight="bold")

        # Style the twin axis
        ax2.tick_params(axis="y", labelsize=10, colors=line_color)
        for spine in ax2.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.8)

        ax2.set_xlim(x.min(), x.max())
        ax.set_xlabel(f"{feature_name} value", fontsize=11, fontweight="bold")

        # Add a thin border
        for spine in ax.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.8)

        return bin_edges_array, fig, target_ratio

    def _get_X_y_shap_with_q_cut(self, feature: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Extracts all X, y pairs and SHAP values that fall within defined quantiles of the feature.

        This method filters the data to include only points within the specified quantile range,
        which helps focus the visualization on the most relevant data points and exclude outliers.

        Args:
            feature (str):
                Feature name to return values for. Must be a column in the X dataframe.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]:
                - x: Selected feature values (filtered by quantile range)
                - y: Target values of selected datapoints
                - shap_val: SHAP values of selected datapoints for the specified feature

        Raises:
            ValueError: If feature is not found in the data.
            RuntimeError: If called before the model is fitted.
        """
        self._check_if_fitted()
        if feature not in self.X.columns:
            raise ValueError(f"Feature '{feature}' not found in data")

        # Prepare arrays
        x = self.X[feature]
        y = self.y

        # Get the feature index for accessing SHAP values
        feature_idx = self.column_names.index(feature)

        # Handle both numpy arrays and pandas DataFrames for shap_values
        if isinstance(self.shap_values, pd.DataFrame):
            shap_val = self.shap_values[feature]
        else:
            # Assume it's a numpy array
            shap_val = self.shap_values[:, feature_idx]

        # Determine quantile ranges for filtering
        x_min = x.quantile(self.min_q)
        x_max = x.quantile(self.max_q)

        # Create filter to include only values within the quantile range
        filter_mask = (x >= x_min) & (x <= x_max)

        # Filter and return data
        return x[filter_mask], y[filter_mask], shap_val[filter_mask]
