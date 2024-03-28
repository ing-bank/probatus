class NotFittedError(Exception):
    """
    Error.
    """

    def __init__(self, message):
        """
        Init error.
        """
        self.message = message


class UnsupportedModelError(Exception):
    """
    Error.
    """

    def __init__(self, message):
        # TODO: Add this check for unsupported models to our implementations.
        """
        Init error.
        """
        self.message = message
