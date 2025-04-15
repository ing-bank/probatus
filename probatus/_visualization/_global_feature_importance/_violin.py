from matplotlib import pyplot as plt
from probatus._visualization._base_plot import DEFAULT_BASE_PARAMS, DEFAULT_BASE_PLOT_PARAMS
from probatus._visualization._global_feature_importance._shap_global_plot import (
    SHAPGlobalPlot,
    DEFAULT_SHAP_GLOBAL_PARAMS,
)
from shap.plots import violin
from probatus.wrapper.shap_new.instance import SHAPInstance
from probatus._common.parameters import get_valid_kwargs
from probatus.wrapper.estimator import BaseModel

DEFAULT_SHAP_VIOLIN_PARAMS: dict = {
    "plot_type": "violin",  # "layered_violin"
    "color": None,  # "blue", "red", ...
}


class ViolinPlot(SHAPGlobalPlot):

    def _create_plot(self) -> plt.Figure:
        if self.plot_kwargs["figsize"] is None:
            height = max(6, 0.5 * self.kwargs["max_display"])
            self.plot_kwargs["figsize"] = (10, height)

        fig = plt.figure(**self.plot_kwargs)

        violin(
            self.shap_instance.explanation,
            show=False,
            **get_valid_kwargs(violin, self.kwargs),
        )

        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        # Init parameters
        kwargs = kwargs | DEFAULT_BASE_PARAMS | DEFAULT_SHAP_GLOBAL_PARAMS | DEFAULT_SHAP_VIOLIN_PARAMS
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, shap_instance, show, **kwargs)
        self.plot_kwargs = plot_kwargs
