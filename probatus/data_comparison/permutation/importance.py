from sklearn.pipeline import Pipeline
from probatus.data_comparison._base import BaseResemblanceModel
from probatus.wrapper import Scorer


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


from packaging.version import parse
from typing import Any, List, Literal, Optional, Union


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
        model: Union[BaseEstimator, Pipeline],
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
            model (Union[BaseEstimator, Pipeline]):
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
            verbose=verbose,
            random_state=random_state,
        )
        self.n_jobs: int = n_jobs
        self.iterations: int = iterations

        # Initialize dataframe to store iteration results
        self.importance_iterations_df: pd.DataFrame = pd.DataFrame(
            {"feature": pd.Series(dtype="object"), "importance": pd.Series(dtype="float64")}
        )

        # Create report dataframe
        self.report_df: pd.DataFrame = pd.DataFrame(
            index=self.data_manager.column_names, columns=["mean_importance", "std_importance"], dtype=float
        )

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
        permutation_result: pd.DataFrame = permutation_importance(
            self.model.estimator,
            self.data_manager.X_test,
            self.data_manager.y_test,
            scoring=self.model.scorer,
            n_repeats=self.iterations,
            n_jobs=self.n_jobs,
        )

        # Process results for each feature
        for feature_index, feature_name in enumerate(self.data_manager.column_names):
            # Store summary statistics
            self.report_df.loc[feature_name, "mean_importance"] = permutation_result["importances_mean"][feature_index]
            self.report_df.loc[feature_name, "std_importance"] = permutation_result["importances_std"][feature_index]

            # Set for all iterations for this feature its importance such that we can
            # calculate the mean and std of the feature importance
            feature_iterations: pd.DataFrame = pd.DataFrame(
                {
                    "feature": np.repeat(feature_name, self.iterations),
                    "importance": permutation_result["importances"][feature_index, :].reshape((self.iterations,)),
                }
            )

            # Append to overall results
            if not feature_iterations.empty:
                self.importance_iterations_df: pd.DataFrame = pd.concat(
                    [self.importance_iterations_df, feature_iterations]
                )

        # Sort features by importance (descending)
        self.report_df.sort_values(by="mean_importance", ascending=False, inplace=True)

        return self

    def plot(self, plot_title: str, top_n: Optional[int] = None, show: bool = False, **plot_kwargs: Any) -> Figure:
        # TODO: Move to plot class
        """
        Plot feature importance as boxplots showing the distribution of importance values.

        This method creates a horizontal boxplot visualization where:
        - Each row represents a feature
        - The boxplot shows the distribution of importance values across iterations
        - Features are sorted by mean importance (most important at the top)
        - Performance metrics are annotated below the plot

        Args:
            plot_title (str): Title of the plot.

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
        self.check_if_computed()

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
            feature_values = self.importance_iterations_df[self.importance_iterations_df["feature"] == feature][
                "importance"
            ]

            # TODO: Remove this once we drop support for matplotlib < 3.10
            # Create horizontal boxplot
            if parse(matplotlib.__version__) >= parse("3.10"):
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
        ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=15)

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
