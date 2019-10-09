class NotFittedError(Exception):
    def __init__(self, message):
        self.message = message


class DimensionalityError(Exception):
    def __init__(self, message):
        self.message = message


class UnsupportedModelError(Exception):
    def __init__(self, message):
        self.message = message