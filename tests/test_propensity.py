from sklearn.datasets import make_classification
from pyrisk.sample_similarity.validation import propensity_check
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_propensity():

    X, y = make_classification(n_samples=10000, n_classes=2, random_state=1)
    X1 = X[y == 0]
    X2 = X[y == 1]
    res = propensity_check(X1, X2, model=RandomForestClassifier(n_estimators=100))
    assert res<= 1
