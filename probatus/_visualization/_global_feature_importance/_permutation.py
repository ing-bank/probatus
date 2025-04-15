from ast import parse
from typing import Any, Dict, List, Tuple

from matplotlib import pyplot as plt
import matplotlib
from probatus._visualization._base_plot import BasePlot, DEFAULT_BASE_PLOT_PARAMS, DEFAULT_BASE_PARAMS
from probatus.wrapper.estimator import BaseModel, BaseScoringModel
import pandas as pd

DEFAULT_PERMUTATION_PARAMS: dict = {
    "max_display": 10,
}


class PermutationPlot(BasePlot):
    def __init__(
        self,
        model: BaseScoringModel,
        feature_report: pd.DataFrame,
        importance_iterations_df: pd.DataFrame,
        show: bool = False,
        **kwargs,
    ) -> plt.Figure:
        # Init parameters
        self._init_parameters(model, show, **kwargs)

        # Prepare environment for plotting
        was_interactive = self._init_environment()

        # Prepare/aggregate/retrieve data
        sorted_features = self._prepare_data(feature_report)

        # Create plot
        fig, ax, boxplot_dicts = self._create_plot(
            importance_iterations_df,
            sorted_features,
        )

        # Apply styling
        self._apply_styling(fig, ax, boxplot_dicts, sorted_features)

        # Show plot (or not)
        self._restore_environment(was_interactive, fig)

        # Return plot
        return fig

    def _init_parameters(self, model: BaseModel, show: bool, **kwargs):
        # Init parameters
        kwargs = kwargs | DEFAULT_BASE_PARAMS | DEFAULT_PERMUTATION_PARAMS
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, show, **kwargs)
        self.plot_kwargs = plot_kwargs

    def _prepare_data(self, feature_report: pd.DataFrame):
        sorted_features = feature_report["mean_importance"].sort_values(ascending=True).index.values

        if self.kwargs["max_display"] is not None and self.kwargs["max_display"] > 0:
            sorted_features = sorted_features[-self.kwargs["max_display"] :]

        return sorted_features

    def _create_plot(
        self, importance_iterations_df: pd.DataFrame, sorted_features: List[str]
    ) -> Tuple[plt.Figure, plt.Axes, List[Dict[str, Any]]]:
        if self.plot_kwargs["figsize"] is None:
            num_features = len(sorted_features)
            height = max(6, 0.5 * num_features)
            self.plot_kwargs["figsize"] = (10, height)

        plt.style.use("default")
        fig, ax = plt.subplots(**self.plot_kwargs)

        boxplot_dicts = []  # Store the dictionaries returned by boxplot
        for position, feature in enumerate(sorted_features):
            feature_values = importance_iterations_df[importance_iterations_df["feature"] == feature]["importance"]

            # TODO: Remove this once we drop support for matplotlib < 3.10
            # Handle matplotlib version differences
            if parse(matplotlib.__version__) >= parse("3.10"):
                box = ax.boxplot(feature_values, positions=[position], orientation="horizontal", patch_artist=True)
            else:
                box = ax.boxplot(feature_values, positions=[position], vert=False, patch_artist=True)

            boxplot_dicts.append(box)

        return fig, ax, boxplot_dicts

    def _apply_styling(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        boxplot_dicts: List[Dict[str, Any]],
        sorted_features: List[str],
    ) -> None:
        # Style the boxplots with SHAP-like colors
        for box in boxplot_dicts:
            for patch in box["boxes"]:
                patch.set_facecolor("#1E88E5")
                patch.set_alpha(0.6)
            for median in box["medians"]:
                median.set_color("#ff0051")
                median.set_linewidth(2)

        # Apply SHAP-like style - set light gray background with white grid
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("white")

        # Add subtle grid lines
        ax.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

        # Set custom tick parameters
        ax.tick_params(axis="both", which="major", labelsize=10)

        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_title(self.kwargs["plot_title"], fontsize=13, fontweight="bold", pad=15)

        # Add performance metrics annotation
        if self.model.scorer:
            ax.annotate(
                self._get_results_text(),
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

    def _get_results_text(self) -> str:
        return (
            f"Train {self.model.scorer.scoring.metric_name}: {round(self.model.train_score, 4)},"
            + f"\nTest {self.model.scorer.scoring.metric_name}: {round(self.model.test_score, 4)}."
        )
