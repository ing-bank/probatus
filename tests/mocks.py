from unittest.mock import Mock


# These are shell classes that define the methods of the models that we use. Each of the functions that we use needs
# To be defined inside these shell classes. Then when we want to write a specific test you need to simply mock.patch
# the desired functionality. You can also set the return_value to the patched method.

class MockClusterer(Mock):
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass

class MockModel(Mock):
    def __init__(self, **kwargs):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass


