import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import Any, List, Optional, Tuple, Union, Literal, cast
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from probatus.utils import BaseFitComputePlotClass, preprocess_data, preprocess_labels, shap_to_df


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
    from probatus.interpret import DependencePlotter

    X, y = make_classification(n_samples=15, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    model = RandomForestClassifier().fit(X, y)
    bdp = DependencePlotter(model)
    shap_values = bdp.fit_compute(X, y)

    bdp.plot(feature=2)
    ```

    <img src="../img/model_interpret_dep.png"/>
    """

    def __init__(self, model: Any, verbose: Literal[0, 1, 2] = 0, random_state: Optional[int] = None) -> None:
        """
        Initializes the DependencePlotter class.

        Args:
            model (Any):
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
        self.model: Any = model
        self.verbose: int = verbose
        self.random_state: Optional[int] = random_state
        self.fitted: bool = False
        self.class_names: List[str] = ["Negative Class", "Positive Class"]

    def __repr__(self) -> str:
        """
        Returns a string representation of the DependencePlotter instance.

        Returns:
            str: String representation of the plotter.
        """
        return f"Shap dependence plotter for {self.model.__class__.__name__}"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        precalc_shap: Optional[pd.DataFrame] = None,
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

            class_names (Optional[List[str]], optional):
                List of class names e.g. ['neg', 'pos']. If None, the default
                ['Negative Class', 'Positive Class'] are used. Default is None.

            precalc_shap (Optional[pd.DataFrame], optional):
                Precalculated SHAP values of shape (n_samples, n_features).
                If provided, they are used directly instead of computing new ones.
                Default is None.

            **shap_kwargs (Any):
                Keyword arguments passed to the SHAP Explainer. Notable parameters include:
                - approximate: If True, uses faster but less accurate SHAP calculation
                - check_additivity: If False, disables the additivity check inside SHAP

        Returns:
            DependencePlotter: The fitted instance with computed SHAP values.

        Raises:
            ValueError: If input data formats are invalid.
        """
        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, index=self.X.index)

        # Set class names with default fallback
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = ["Negative Class", "Positive Class"]

        self.shap_vals_df = shap_to_df(
            self.model,
            self.X,
            precalc_shap=precalc_shap,
            verbose=self.verbose,
            random_state=self.random_state,
            **shap_kwargs,
        )

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
        return self.shap_vals_df

    def fit_compute(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        column_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
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

            class_names (Optional[List[str]], optional):
                List of class names e.g. ['neg', 'pos']. If None, the default
                ['Negative Class', 'Positive Class'] are used. Default is None.

            precalc_shap (Optional[pd.DataFrame], optional):
                Precalculated SHAP values of shape (n_samples, n_features).
                If provided, they are used directly instead of computing new ones.
                Default is None.

            **shap_kwargs (Any):
                Keyword arguments passed to the SHAP Explainer. Notable parameters include:
                - approximate: If True, uses faster but less accurate SHAP calculation
                - check_additivity: If False, disables the additivity check inside SHAP

        Returns:
            pd.DataFrame: DataFrame containing SHAP values for each feature in X.
                          Each column corresponds to a feature, and each row to an observation.

        Raises:
            ValueError: If input data formats are invalid.
            RuntimeError: If internal computation fails.
        """
        self.fit(X, y, column_names=column_names, class_names=class_names, precalc_shap=precalc_shap, **shap_kwargs)
        return self.compute()

    def plot(
        self,
        feature: Union[str, int],
        figsize: Tuple[float, float] = (15, 10),
        bins: Union[int, List[float]] = 10,
        show: bool = True,
        min_q: float = 0,
        max_q: float = 1,
        alpha: float = 1.0,
    ) -> List[Axes]:
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
                Default is True.

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
            List[matplotlib.axes.Axes]: List of axes that include the plots.
                                        [0] - SHAP dependence plot
                                        [1] - Feature distribution and target rate plot

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
        if isinstance(feature, int):
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
        _: Figure = plt.figure(1, figsize=figsize)
        ax1: Axes = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2: Axes = plt.subplot2grid((3, 1), (2, 0))

        self._dependence_plot(feature=feature_name, ax=ax1)
        self._target_rate_plot(feature=feature_name, bins=bins, ax=ax2)

        # Ensure both plots have the same x-axis limits for visual consistency
        ax2.set_xlim(ax1.get_xlim())

        if show:
            plt.show()
        else:
            plt.close()

        return [ax1, ax2]

    def _dependence_plot(self, feature: str, ax: Optional[Axes] = None) -> Axes:
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
            matplotlib.axes.Axes: Axes on which plot is drawn.

        Raises:
            ValueError: If feature is not found in the dataset.
            RuntimeError: If called before the model is fitted.
        """
        if isinstance(feature, int):
            feature = self.column_names[feature]

        # Get filtered data based on quantile range
        X, y, shap_val = self._get_X_y_shap_with_q_cut(feature=feature)

        # Create or use provided axes
        ax = plt.gca() if ax is None else cast(Axes, ax)

        # Create scatter plot for negative class (y=0) &  (y=1)
        ax.scatter(X[y == 0], shap_val[y == 0], label=self.class_names[0], color="lightblue", alpha=self.alpha)
        ax.scatter(X[y == 1], shap_val[y == 1], label=self.class_names[1], color="darkred", alpha=self.alpha)
        ax.set_ylabel("Shap value")
        ax.set_title(f"Dependence plot for {feature} feature")
        ax.legend()

        return ax

    def _target_rate_plot(
        self, feature: Union[str, int], bins: Union[int, List[float]] = 10, ax: Optional[Axes] = None
    ) -> Tuple[Union[List[float], np.ndarray], Axes, pd.Series]:
        """
        Plots the distribution of the feature values and the target rate as function of the feature.

        This creates a histogram of feature values with a line showing the target rate
        (proportion of positive class) for each bin of feature values.

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
            Tuple[Union[List[float], np.ndarray], matplotlib.axes.Axes, pd.Series]:
                - Boundaries of bins used for the histogram
                - Axis on which plot is drawn
                - Series containing target ratio (proportion of positive class) for each bin

        Raises:
            ValueError: If feature is not found in the dataset.
            RuntimeError: If called before the model is fitted.
            TypeError: If bins is not an integer or a list of floats.
        """
        # Create or use provided axes
        ax = plt.gca() if ax is None else cast(Axes, ax)

        # Handle feature name extraction
        feature_name: str = (
            self.column_names[feature]
            if isinstance(feature, int) and isinstance(self.column_names, list)
            else str(feature)
        )

        # Get filtered data based on quantile range
        x, y, _ = self._get_X_y_shap_with_q_cut(feature=feature_name)

        # Create bins if not explicitly supplied
        bin_edges: Union[List[float], np.ndarray]
        if isinstance(bins, int):
            # Use KBinsDiscretizer to create uniform bins
            simple_binner = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="uniform").fit(
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
        target_ratio = dfs["y"].mean()  # Proportion of positive class in each bin
        x_vals = dfs[feature_name].mean()  # Mean feature value in each bin

        # Transform the first and last bin to work with plt.hist method
        # Replace infinity values with actual min/max for plotting
        bin_edges_for_hist = bin_edges_array.copy()
        if bin_edges_for_hist[0] == -np.inf:
            bin_edges_for_hist[0] = x.min()
        if bin_edges_for_hist[-1] == np.inf:
            bin_edges_for_hist[-1] = x.max()

        # Plot histogram of feature values
        ax.hist(x, bins=cast(Union[int, List[float]], bin_edges_for_hist), lw=2, alpha=0.4)
        ax.set_ylabel("Counts")

        # Create twin axis for target rate line
        ax2 = ax.twinx()
        ax2 = cast(Axes, ax2)
        ax2.plot(x_vals, target_ratio, color="red")
        ax2.set_ylabel("Target rate", color="red", fontsize=12)
        ax2.set_xlim(x.min(), x.max())
        ax.set_xlabel(f"{feature_name} feature values")

        return bin_edges_array, ax, target_ratio

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
        shap_val = self.shap_vals_df[feature]

        # Determine quantile ranges for filtering
        x_min = x.quantile(self.min_q)
        x_max = x.quantile(self.max_q)

        # Create filter to include only values within the quantile range
        filter_mask = (x >= x_min) & (x <= x_max)

        # Filter and return data
        return x[filter_mask], y[filter_mask], shap_val[filter_mask]
