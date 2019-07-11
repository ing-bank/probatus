import unittest

class Test(unittest.TestCase):

    def test_propensity(self):

        from sklearn.datasets import make_classification
        from pyrisk.validation import propensity_check
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        X, y = make_classification(n_samples=10000, n_classes=2, random_state=1)
        X1 = X[y == 0]
        X2 = X[y == 1]
        res = propensity_check(X1, X2, model=RandomForestClassifier(n_estimators=100))
        self.assertLessEqual(res, 1)


if __name__ == "__main__":

    unittest.main(verbosity=2)
