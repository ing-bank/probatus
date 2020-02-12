import pandas as pd

class MockClusterer():
    def __init__(self, num_clusters = 3, **kwargs):
        self.num_clusters = 3

    def fit(self, X):
        return self

    def predict(self, X):
        output = []
        for index in range(len(X)):
            output.append(index % self.num_clusters)
        return output

class MockModel():
    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        return self


