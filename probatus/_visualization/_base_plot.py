from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union, List

from matplotlib import pyplot as plt

from probatus._visualization.enum import PlotTypeEnum
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel
from probatus.wrapper.shap_new.instance import SHAPInstance


class BasePlot(ABC):
    def __init__(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        show: bool = False,
        **kwargs,
    ):
        # TODO: Perhaps distinguish between different type of available kwargs per step
        self.model = model
        self.data_manager = data_manager
        self.show = show
        self.kwargs = kwargs

    @abstractmethod
    def plot(
        self,
        shap_instance: SHAPInstance,
        plot_type: PlotTypeEnum,
        split_selection: Literal["full", "train", "test"] = "test",
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        show: bool = False,
        **kwargs,
    ) -> plt.Figure | List[plt.Figure]:
        # TODO: Perhaps distinguish between different type of available kwargs per step

        # Prepare data for plotting
        # Prepare styling
        # Create plot
        # Show plot (or not)
        # Return plot
        pass

    def _prepare_environment(self) -> bool:
        # Setup plotting environment
        was_interactive = plt.isinteractive()
        plt.ioff()

        return was_interactive

    @abstractmethod
    def _prepare_data(self):
        pass

    @abstractmethod
    def _create_plot(self):
        pass

    @abstractmethod
    def _apply_styling(self):
        pass

    @abstractmethod
    def _create_labels(self):
        pass

    def _restore_environment(self, was_interactive: bool, fig: plt.Figure) -> None:
        # Finalize and handle display
        plt.tight_layout()
        if self.show:
            plt.show(block=False)
        else:
            plt.close(fig)

        # Restore interactive state
        if was_interactive:
            plt.ion()
