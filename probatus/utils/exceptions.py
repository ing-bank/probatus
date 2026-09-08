from __future__ import annotations


class NotFittedError(Exception):
    """
    Error.
    """

    def __init__(self, message: str) -> None:
        """
        Init error.
        """
        self.message = message
