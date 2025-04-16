from matplotlib import pyplot as plt
from probatus._visualization._base_plot import DEFAULT_BASE_PARAMS, DEFAULT_BASE_PLOT_PARAMS
from probatus._visualization._local_explanation._shap_local_plot import SHAPLocalPlot, DEFAULT_SHAP_LOCAL_PARAMS
from shap.plots import heatmap
from probatus.wrapper.shap_new.instance import SHAPInstance
from probatus._common.parameters import get_valid_kwargs
from probatus.wrapper.estimator import BaseModel

DEFAULT_SHAP_HEATMAP_PARAMS: dict = {
    "order": None,  # "abs_mean", "abs_max"
}


class HeatmapPlot(SHAPLocalPlot):

    def _create_plot(self) -> plt.Figure:
        if self.plot_kwargs["figsize"] is None:
            height = max(6, 0.5 * self.kwargs["max_display"])
            self.plot_kwargs["figsize"] = (10, height)

        fig = plt.figure(**self.plot_kwargs)

        heatmap(
            self.shap_instance.explanation,
            show=False,
            **get_valid_kwargs(heatmap, self.kwargs),
        )

        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        # Init parameters
        kwargs = DEFAULT_BASE_PARAMS | DEFAULT_SHAP_LOCAL_PARAMS | DEFAULT_SHAP_HEATMAP_PARAMS | kwargs
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, shap_instance, show, **kwargs)
        self.plot_kwargs = plot_kwargs

        # Verify the parameters
        self._verify_parameters()

    def _verify_parameters(self):
        # Verify the order parameter
        if self.kwargs["order"] is not None:

            if self.kwargs["order"] not in ["abs_mean", "abs_max"]:
                raise ValueError(f"Invalid order: {self.kwargs['order']}")

            elif self.kwargs["order"] == "abs_mean":
                self.kwargs["feature_values"] = self.shap_instance.explanation.abs.mean(0)

            elif self.kwargs["order"] == "abs_max":
                self.kwargs["feature_values"] = self.shap_instance.explanation.abs.max(0)
