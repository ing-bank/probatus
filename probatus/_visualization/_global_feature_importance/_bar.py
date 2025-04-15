from matplotlib import pyplot as plt
from probatus._visualization._base_plot import DEFAULT_BASE_PARAMS, DEFAULT_BASE_PLOT_PARAMS
from probatus._visualization._global_feature_importance._shap_global_plot import (
    SHAPGlobalPlot,
    DEFAULT_SHAP_GLOBAL_PARAMS,
)
from shap.plots import bar
from probatus.wrapper.shap_new.instance import SHAPInstance
from probatus._common.parameters import get_valid_kwargs
from probatus.wrapper.estimator import BaseModel

DEFAULT_SHAP_BAR_PARAMS: dict = {
    "color": None,  # "blue", "red", ...
    "cohort": None,
    "sample_index": None,
    "order": None,  # "abs_mean", "abs_max"
}


class BarPlot(SHAPGlobalPlot):

    def _create_plot(self) -> plt.Figure:
        if self.plot_kwargs["figsize"] is None:
            height = max(6, 0.5 * self.kwargs["max_display"])
            self.plot_kwargs["figsize"] = (10, height)

        fig = plt.figure(**self.plot_kwargs)

        # Either sample index or cohort must be specified
        if self.kwargs["sample_index"] is not None:
            bar(
                self.shap_instance.explanation[self.kwargs.get("sample_index")],
                show=False,
                **get_valid_kwargs(bar, self.kwargs),
            )

        elif self.kwargs["cohort_aggregation"] is not None:
            bar(
                self.kwargs["cohort_aggregation"],
                show=False,
                **get_valid_kwargs(bar, self.kwargs),
            )

        else:
            bar(
                self.shap_instance.explanation,
                show=False,
                **get_valid_kwargs(bar, self.kwargs),
            )

        return fig

    def _init_parameters(self, model: BaseModel, shap_instance: SHAPInstance, show: bool, **kwargs):
        # Init parameters
        kwargs = kwargs | DEFAULT_BASE_PARAMS | DEFAULT_SHAP_GLOBAL_PARAMS | DEFAULT_SHAP_BAR_PARAMS
        plot_kwargs = DEFAULT_BASE_PLOT_PARAMS

        # Add some specific kwargs for permutation plots
        super().__init__(model, shap_instance, show, **kwargs)
        self.plot_kwargs = plot_kwargs

        # Verify the parameters
        self._verify_parameters()

    def _verify_parameters(self):
        # Verify the cohort parameter
        if self.kwargs["cohort"] is not None:
            if not isinstance(self.kwargs["cohort"], int):
                raise TypeError(f"Cohort must be an integer, got {type(self.kwargs['cohort'])}")

            elif self.kwargs["order"] not in ["abs_mean", "abs_max"]:
                raise ValueError(f"Invalid order: {self.kwargs['order']}")

            elif self.kwargs["order"] == "abs_mean":
                self.kwargs["cohort_aggregation"] = self.shap_instance.explanation.cohorts(
                    self.kwargs["cohort"]
                ).abs.mean(0)

            elif self.kwargs["order"] == "abs_max":
                self.kwargs["cohort_aggregation"] = self.shap_instance.explanation.cohorts(
                    self.kwargs["cohort"]
                ).abs.max(0)

        # Verify the order parameter
        if self.kwargs["order"] is not None:

            if self.kwargs["order"] not in ["abs_mean", "abs_max"]:
                raise ValueError(f"Invalid order: {self.kwargs['order']}")

            elif self.kwargs["order"] == "abs_mean":
                self.kwargs["order"] = self.shap_instance.explanation.abs.mean(0)

            elif self.kwargs["order"] == "abs_max":
                self.kwargs["order"] = self.shap_instance.explanation.abs.max(0)

        # Verify the sample_index parameter
        if self.kwargs["sample_index"] is not None:
            if not isinstance(self.kwargs["sample_index"], int):
                raise TypeError(f"Sample index must be an integer, got {type(self.kwargs['sample_index'])}")

            elif (
                self.kwargs["sample_index"] < 0
                or self.kwargs["sample_index"] >= self.shap_instance.explanation.data.shape[0]
            ):
                raise ValueError(
                    f"Sample index must be between 0 and {self.shap_instance.explanation.data.shape[0] - 1}, got {self.kwargs['sample_index']}"
                )
