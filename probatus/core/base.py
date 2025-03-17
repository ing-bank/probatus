from abc import ABC, abstractmethod
from typing import Any

from probatus.core import NotFittedError


class BaseFitComputeClass(ABC):
    """
    Abstract base class that defines the interface for classes with fit and compute functionality.

    This class establishes a common interface for objects that need to:
    1. Be fitted to data (fit method)
    2. Compute results based on that fitting (compute method)
    3. Do both operations in sequence (fit_compute method)

    All subclasses must implement these three methods.
    """

    # Flag to track if the instance has been fitted
    fitted: bool = False

    def _check_if_fitted(self) -> None:
        """
        Checks if the object has been fitted.

        Raises:
            NotFittedError: If the object has not been fitted yet.
        """
        if not self.fitted:
            raise NotFittedError("This estimator is not fitted yet. Call 'fit' before using this method.")

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> "BaseFitComputeClass":
        """
        Fit the estimator to data.

        This method must be implemented by all subclasses. Typically, it should:
        1. Process the input data
        2. Set internal parameters based on that data
        3. Set self.fitted = True
        4. Return self for method chaining

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self: The fitted instance.
        """
        pass

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute results based on the fitted estimator.

        This method must be implemented by all subclasses. It should:
        1. Check if the estimator is fitted
        2. Compute results based on the internal state and provided arguments

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The computed results.
        """
        pass

    @abstractmethod
    def fit_compute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fit the estimator and compute results in a single step.

        This method must be implemented by all subclasses. A typical implementation
        would call fit() followed by compute(), but subclasses may optimize this
        process for better performance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The computed results.
        """
        pass


class BaseFitComputePlotClass(BaseFitComputeClass):
    """
    Abstract base class that extends BaseFitComputeClass with plotting functionality.

    This class adds a plotting interface to the fit-compute pattern, allowing
    subclasses to visualize their results. All subclasses must implement the
    plot method in addition to the methods required by BaseFitComputeClass.
    """

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """
        Visualize the results of the computation.

        This method must be implemented by all subclasses. It should:
        1. Check if the estimator is fitted
        2. Generate and return a visualization based on the computed results

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The plot object or visualization result.
        """
        pass
