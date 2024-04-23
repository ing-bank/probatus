class NotFittedError(Exception):
    """
    Error.
    """

    def __init__(self, message):
        """
        Init error.
        """
        self.message = message
