from matplotlib import pyplot as plt
from probatus._visualization._base_plot import DEFAULT_BASE_PARAMS, DEFAULT_BASE_PLOT_PARAMS
from probatus._visualization._local_explanation._shap_local_plot import SHAPLocalPlot, DEFAULT_SHAP_LOCAL_PARAMS
from shap.plots import scatter
from probatus.wrapper.shap_new.instance import SHAPInstance
from probatus._common.parameters import get_valid_kwargs
from probatus.wrapper.estimator import BaseModel

DEFAULT_SHAP_SCATTER_PARAMS: dict = {
    "color": None,  # "blue", "red", ... (or Explanation object)
    "feature_name": None,
    "color_based_on_feature_shap_values": None,
}


class ScatterPlot(SHAPLocalPlot):

    def _create_plot(self) -> plt.Figure:
        if self.plot_kwargs["figsize"] is None:
            height = max(6, 0.5 * self.kwargs["max_display"])
            self.plot_kwargs["figsize"] = (10, height)

        fig = plt.figure(**self.plot_kwargs)

        if self.kwargs.get("feature_name") is not None:
            scatter(
                self.shap_instance.explanation[:, self.kwargs["feature_name"]],
                show=False,
                **get_valid_kwargs(scatter, self.kwargs),
            )
        else:
            scatter(
                self.shap_instance.explanation[:, self.shap_instance.explanation.abs.mean(0).argsort[-1]],
                show=False,
                **get_valid_kwargs(scatter, self.kwargs),
            )

        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        # Init parameters
        kwargs = DEFAULT_BASE_PARAMS | DEFAULT_SHAP_LOCAL_PARAMS | DEFAULT_SHAP_SCATTER_PARAMS | kwargs
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, shap_instance, show, **kwargs)
        self.plot_kwargs = plot_kwargs

        # Verify the parameters
        self._verify_parameters()

    def _verify_parameters(self):
        # Verify the color parameter
        if self.kwargs.get("color") is not None:
            if not isinstance(self.kwargs["color"], str):
                raise TypeError(f"Color must be a string, got {type(self.kwargs['color'])}")
        else:
            if self.kwargs.get("color_based_on_feature_shap_values") is not None:
                if (
                    isinstance(self.kwargs.get("color_based_on_feature_shap_values"), bool)
                    and self.kwargs.get("color_based_on_feature_shap_values") is True
                ):
                    self.kwargs["color"] = self.shap_instance.explanation
                elif isinstance(self.kwargs.get("color_based_on_feature_shap_values"), str):
                    self.kwargs["color"] = self.shap_instance.explanation[
                        :, self.kwargs.get("color_based_on_feature_shap_values")
                    ]
                else:
                    raise TypeError(
                        f"Color based on feature shap values must be a boolean or a string, got {type(self.kwargs['color_based_on_feature_shap_values'])}"
                    )
