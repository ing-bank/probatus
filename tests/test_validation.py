from sklearn.datasets import make_classification
from pyrisk.sample_similarity import validation as validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def test_propensity():
    X, y = make_classification(n_samples=10000,
                               n_classes=2,
                               random_state=1)
    X1 = X[y == 0]
    X2 = X[y == 1]
    res, feat = validation.propensity_check(X1, X2)

    X_2, y_2 = make_classification(n_samples=10000,
                                   n_classes=2,
                                   random_state=1000)
    X3 = X_2[y_2 == 1]
    X4 = X_2[y_2 == 0]
    res2, feat2 = validation.propensity_check(X3, X4)
    assert ((res - res2) <= 0.05)

def test_feature_importance():

    X, y = make_classification(n_samples=10000, n_classes=2, random_state=1)
    model = RandomForestClassifier()
    model.fit(X,y)
    feat_importance = validation.get_feature_importance(model)
    assert (X.shape[1] == feat_importance.shape[0])
