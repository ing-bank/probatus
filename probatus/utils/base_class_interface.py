from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, ParamSpec, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from probatus.utils import NotFittedError

FitParams = ParamSpec("FitParams")
ComputeParams = ParamSpec("ComputeParams")
Result = TypeVar("Result")


class BaseFitComputeClass(ABC, Generic[FitParams, ComputeParams, Result]):
    """
    Placeholder that must be overwritten by subclass.
    """

    fitted = False

    def _check_if_fitted(self) -> None:
        """
        Checks if object has been fitted. If not, NotFittedError is raised.
        """
        if not self.fitted:
            raise (NotFittedError("The object has not been fitted. Please run fit() method first"))

    @abstractmethod
    def fit(self, *args: FitParams.args, **kwargs: FitParams.kwargs) -> object:
        """
        Placeholder that must be overwritten by subclass.
        """
        pass

    @abstractmethod
    def compute(self, *args: ComputeParams.args, **kwargs: ComputeParams.kwargs) -> Result:
        """
        Placeholder that must be overwritten by subclass.
        """
        pass

    @abstractmethod
    def fit_compute(self, *args: FitParams.args, **kwargs: FitParams.kwargs) -> Result:
        """
        Placeholder that must be overwritten by subclass.
        """
        pass


class BaseFitComputePlotClass(BaseFitComputeClass[FitParams, ComputeParams, Result]):
    """
    Base class.
    """

    @abstractmethod
    def plot(self, *args: FitParams.args, **kwargs: FitParams.kwargs) -> Figure | Axes | list[Axes] | list[list[Axes]]:
        """
        Placeholder method for plotting.
        """
        pass
