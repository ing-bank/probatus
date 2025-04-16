from matplotlib import pyplot as plt
from probatus._common.parameters import get_valid_kwargs
from probatus._visualization._base_plot import DEFAULT_BASE_PARAMS, DEFAULT_BASE_PLOT_PARAMS
from probatus._visualization._global_feature_importance._shap_global_plot import SHAPGlobalPlot
from shap.plots import beeswarm
from probatus._visualization._global_feature_importance._shap_global_plot import DEFAULT_SHAP_GLOBAL_PARAMS
from probatus.wrapper.estimator import BaseModel
from probatus.wrapper.shap_new.instance import SHAPInstance

DEFAULT_SHAP_BEE_SWARM_PARAMS: dict = {
    "color": None,  # "blue", "red", ...
    "cohort": None,
    "sample_index": None,
    "order": None,  # "abs_mean", "abs_max"
}


class BeeswarmPlot(SHAPGlobalPlot):

    def _create_plot(self) -> plt.Figure:
        if self.plot_kwargs["figsize"] is None:
            height = max(6, 0.5 * self.kwargs["max_display"])
            self.plot_kwargs["figsize"] = (10, height)

        fig = plt.figure(**self.plot_kwargs)

        # Either sample index or cohort must be specified
        if self.kwargs["sample_index"] is not None:
            beeswarm(
                self.shap_instance.explanation[self.kwargs.get("sample_index")],
                show=False,
                **get_valid_kwargs(beeswarm, self.kwargs),
            )

        elif self.kwargs["cohort_aggregation"] is not None:
            beeswarm(
                self.kwargs["cohort_aggregation"],
                show=False,
                **get_valid_kwargs(beeswarm, self.kwargs),
            )

        else:
            beeswarm(
                self.shap_instance.explanation,
                show=False,
                **get_valid_kwargs(beeswarm, self.kwargs),
            )

        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        # Init parameters
        kwargs = DEFAULT_BASE_PARAMS | DEFAULT_SHAP_GLOBAL_PARAMS | DEFAULT_SHAP_BEE_SWARM_PARAMS | kwargs
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, shap_instance, show, **kwargs)
        self.plot_kwargs = plot_kwargs

        # Verify the parameters
        self._verify_parameters()
