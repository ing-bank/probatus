class ApproximationWarning(Warning):
    def __init__(self, message):
        self.message = message


class NotIntendedUseWarning(Warning):
    def __init__(self, message):
        self.message = message