from typing import Any, Dict, List, Optional, Union, Literal

from matplotlib import pyplot as plt
from probatus._common.parameters import extract_parameters
from probatus._visualization._base_plot import BasePlot, DEFAULT_BASE_PLOT_PARAMS, DEFAULT_BASE_AGGREGATION_PARAMS
from probatus._visualization.enum import PlotTypeEnum
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel
from probatus.wrapper.shap_new.instance import SHAPInstance
import pandas as pd

DEFAULT_PERMUTATION_PLOT_PARAMS: dict = {
    "top_n": 10,
}


class PermutationPlot(BasePlot):
    def __init__(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        show: bool = False,
        **kwargs,
    ):
        # Add some specific kwargs for permutation plots
        super().__init__(model, data_manager, show, **kwargs)

    def plot(
        self,
        plot_type: PlotTypeEnum,
        split_selection: Literal["full", "train", "test"] = "test",
        feature_report: pd.DataFrame = pd.DataFrame(),
        importance_iterations_df: pd.DataFrame = pd.DataFrame(),
        show: bool = False,
        **kwargs,
    ) -> plt.Figure | List[plt.Figure]:
        # TODO: Perhaps distinguish between different type of available kwargs per step
        # Filter out various kwargs properties
        kwargs, plot_params, aggregation_params, permutation_params = extract_parameters(
            kwargs=kwargs,
            default_kwarg_dicts=[
                DEFAULT_BASE_PLOT_PARAMS,
                DEFAULT_BASE_AGGREGATION_PARAMS,
                DEFAULT_PERMUTATION_PLOT_PARAMS,
            ],
        )

        #

        # Prepare environment for plotting
        was_interactive = self._init_environment()

        # Prepare data
        self._prepare_data(feature_report=feature_report, top_n=top_n)

        # Create plot
        self._create_plot()

        # Apply styling
        self._apply_styling()

        # Show plot (or not) # TODO: pass fig
        self._restore_environment(was_interactive=was_interactive, fig=None)

        # Return plot
        pass

    def _prepare_data(self, feature_report: pd.DataFrame, top_n: Optional[int] = None):
        sorted_features = feature_report["mean_importance"].sort_values(ascending=True).index.values

        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[-top_n:]

        return sorted_features

    def _create_plot(self):
        pass

    def _apply_styling(self):
        pass

    def _create_labels(self):
        pass
