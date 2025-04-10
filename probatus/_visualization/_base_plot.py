from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union, List

from matplotlib import pyplot as plt

from probatus._visualization.enum import PlotTypeEnum
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel
from probatus.wrapper.shap_new.instance import SHAPInstance
from probatus.wrapper.shap_new.manager import SHAPManager

DEFAULT_BASE_PLOT_PARAMS: dict = {
    "plot_title": None,
    "figsize": None,
}

DEFAULT_BASE_AGGREGATION_PARAMS: dict = {
    "class_selection": None,
    "weights": None,
    "multi_class_aggregation": None,
    "shap_variance_penalty_factor": 0,
}


class BasePlot(ABC):
    def __init__(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        shap_manager: SHAPManager,
        show: bool = False,
        **kwargs,
    ):
        # TODO: Perhaps distinguish between different type of available kwargs per step
        self.model = model
        self.data_manager = data_manager
        self.shap_manager = shap_manager
        self.show = show
        self.kwargs = kwargs

    @abstractmethod
    def plot(
        self,
        plot_type: PlotTypeEnum,
        split_selection: Literal["full", "train", "test"] = "test",
        show: bool = False,
        **kwargs,
    ) -> plt.Figure | List[plt.Figure]:
        # Load right SHAP data

        # verify all params required are given

        # Prepare data for plotting

        # Prepare styling

        # Create plot

        # Show plot (or not)

        # Return plot
        pass

    def _init_environment(self) -> bool:
        # Setup plotting environment & prevent always showing figure
        # so that it should only be shown when requested.
        was_interactive = plt.isinteractive()
        plt.ioff()

        return was_interactive

    @abstractmethod
    def _prepare_data(self, *args: Any):
        pass

    @abstractmethod
    def _create_plot(self, *args: Any):
        pass

    @abstractmethod
    def _apply_styling(self, *args: Any):
        pass

    @abstractmethod
    def _create_labels(self, *args: Any):
        pass

    def _restore_environment(self, was_interactive: bool, fig: plt.Figure) -> None:
        # Finalize and handle display
        plt.tight_layout()
        if self.show:
            plt.show(block=False)
        else:
            plt.close(fig)

        # Restore previous plotting and environment showing figure settings.
        if was_interactive:
            plt.ion()
