class NotFittedError(Exception):
    """
    Exception raised when a method is called on an estimator that has not been fitted yet.

    This error is typically raised when a method that requires a fitted model
    (such as predict, transform, or score) is called before the fit method.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the NotFittedError with a descriptive message.

        Parameters
        ----------
        message : str
            A descriptive error message explaining which estimator or method
            was called before fitting.

        Examples
        --------
        >>> raise NotFittedError("This estimator is not fitted yet. Call 'fit' before using this method.")
        """
        self.message = message

        super().__init__(message)
