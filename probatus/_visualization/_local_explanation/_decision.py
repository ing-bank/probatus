from matplotlib import pyplot as plt
from probatus._visualization._base_plot import DEFAULT_BASE_PARAMS, DEFAULT_BASE_PLOT_PARAMS
from probatus._visualization._local_explanation._shap_local_plot import SHAPLocalPlot, DEFAULT_SHAP_LOCAL_PARAMS
from shap.plots import decision as decision_plot
from probatus.wrapper.shap_new.instance import SHAPInstance
from probatus._common.parameters import get_valid_kwargs
from probatus.wrapper.estimator import BaseModel

DEFAULT_SHAP_DECISION_PARAMS: dict = {
    "plot_color": None,  # "blue", "red", ...
    "sample_index": 0,
}


class DecisionPlot(SHAPLocalPlot):

    def _create_plot(self) -> plt.Figure:
        if self.plot_kwargs["figsize"] is None:
            height = max(6, 0.5 * self.kwargs["max_display"])
            self.plot_kwargs["figsize"] = (10, height)

        fig = plt.figure(**self.plot_kwargs)

        decision_plot(
            self.shap_instance.explanation[self.kwargs.get("sample_index")],
            show=False,
            **get_valid_kwargs(decision_plot, self.kwargs),
        )

        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        # Init parameters
        kwargs = DEFAULT_BASE_PARAMS | DEFAULT_SHAP_LOCAL_PARAMS | DEFAULT_SHAP_DECISION_PARAMS | kwargs
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, shap_instance, show, **kwargs)
        self.plot_kwargs = plot_kwargs

        # Verify the parameters
        self._verify_parameters()

    def _verify_parameters(self):
        # Verify number of features displayed
        if self.kwargs["max_display"] is not None:
            if not isinstance(self.kwargs["max_display"], int):
                raise TypeError(f"Max display must be an integer, got {type(self.kwargs['max_display'])}")

            # max_display not supported by decision plot.
            if self.kwargs["feature_display_range"] is None:
                self.kwargs["feature_display_range"] = self.kwargs["max_display"]

        # Verify the sample_index parameter
        if self.kwargs["sample_index"] is not None:
            if not isinstance(self.kwargs["sample_index"], (int, list[int])):
                raise TypeError(
                    f"Sample index must be an integer or a list of integers, got {type(self.kwargs['sample_index'])}"
                )

            if isinstance(self.kwargs["sample_index"], list[int]):
                for index in self.kwargs["sample_index"]:
                    if index < 0 or index >= self.shap_instance.explanation.data.shape[0]:
                        raise ValueError(
                            f"Sample index must be between 0 and {self.shap_instance.explanation.data.shape[0] - 1}, got {self.kwargs['sample_index']}"
                        )
            else:
                if (
                    self.kwargs["sample_index"] < 0
                    or self.kwargs["sample_index"] >= self.shap_instance.explanation.data.shape[0]
                ):
                    raise ValueError(
                        f"Sample index must be between 0 and {self.shap_instance.explanation.data.shape[0] - 1}, got {self.kwargs['sample_index']}"
                    )
