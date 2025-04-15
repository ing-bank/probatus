from abc import ABC, abstractmethod
from typing import Any

from matplotlib import pyplot as plt

from probatus.wrapper.estimator import BaseModel

DEFAULT_BASE_PLOT_PARAMS: dict = {
    "figsize": None,
}

DEFAULT_BASE_PARAMS: dict = {
    "plot_title": None,
}


class BasePlot(ABC):
    def __init__(
        self,
        model: BaseModel,
        show: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.show = show
        self.kwargs = kwargs

    @abstractmethod
    def _init_parameters(self, *args: Any, **kwargs: Any):
        pass

    def _init_environment(self) -> bool:
        # Setup plotting environment & prevent always showing figure
        # so that it should only be shown when requested.
        was_interactive = plt.isinteractive()
        plt.ioff()

        return was_interactive

    @abstractmethod
    def _create_plot(self, *args: Any):
        pass

    @abstractmethod
    def _apply_styling(self, *args: Any):
        pass

    def _restore_environment(self, was_interactive: bool, fig: plt.Figure) -> None:
        if self.show:
            plt.show(block=False)
        else:
            plt.close(fig)

        # Restore previous plotting and environment showing figure settings.
        if was_interactive:
            plt.ion()
