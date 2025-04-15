from typing import Union
from matplotlib import pyplot as plt
from probatus._visualization._base_plot import BasePlot
from probatus.wrapper.estimator import BaseModel, BaseScoringModel
from probatus.wrapper.shap_new.instance import SHAPInstance

DEFAULT_SHAP_GLOBAL_PARAMS: dict = {
    "max_display": 10,
    "plot_title": "SHAP Global Feature Importance",
}


class SHAPGlobalPlot(BasePlot):
    def __init__(
        self,
        model: Union[BaseModel, BaseScoringModel],
        shap_instance: SHAPInstance,
        show: bool = False,
        **kwargs,
    ) -> None:
        # Init parameters
        self._init_parameters(model, shap_instance, show, **kwargs)

        # Prepare environment for plotting
        was_interactive = self._init_environment()

        # Create plot
        fig = self._create_plot()

        # Apply styling
        self._apply_styling(fig)

        # Show plot (or not)
        self._restore_environment(was_interactive, fig)

        # Return plot
        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        super().__init__(model, show, **kwargs)
        self.shap_instance = shap_instance

    def _apply_styling(
        self,
        fig: plt.Figure,
    ) -> None:
        ax = plt.gca()

        # Apply styling
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)

        plt.suptitle(self.kwargs["plot_title"], fontsize=13, fontweight="bold", y=1.02)

        # Add performance metrics annotation
        if self.model.scorer:
            plt.figtext(
                0.5,
                -0.05,
                self._get_results_text(),
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

        # Finalize and handle display
        plt.tight_layout()

    def _get_results_text(self) -> str:
        return (
            f"Train {self.model.scorer.scoring.metric_name}: {round(self.model.train_score, 4)},"
            + f"\nTest {self.model.scorer.scoring.metric_name}: {round(self.model.test_score, 4)}."
        )
