from abc import ABC, abstractmethod

from probatus.utils import NotFittedError


class BaseFitComputeClass(ABC):
    """
    Placeholder that must be overwritten by subclass.
    """

    fitted = False

    def _check_if_fitted(self):
        """
        Checks if object has been fitted. If not, NotFittedError is raised.
        """
        if not self.fitted:
            raise (NotFittedError("The object has not been fitted. Please run fit() method first"))

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Placeholder that must be overwritten by subclass.
        """
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Placeholder that must be overwritten by subclass.
        """
        pass

    @abstractmethod
    def fit_compute(self, *args, **kwargs):
        """
        Placeholder that must be overwritten by subclass.
        """
        pass


class BaseFitComputePlotClass(BaseFitComputeClass):
    """
    Base class.
    """

    @abstractmethod
    def plot(self, *args, **kwargs):
        """
        Placeholder method for plotting.
        """
        pass
