from sklearn.datasets import make_classification
from pyrisk.sample_similarity import validation as validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_propensity():

    X, y = make_classification(n_samples=10000, n_classes=2, random_state=1)
    X1 = X[y == 0]
    X2 = X[y == 1]
    res, feat = validation.propensity_check(X1, X2)
    assert (res<= 1)

def test_feature_importance():
    
    X, y = make_classification(n_samples=10000, n_classes=2, random_state=1)
    model = RandomForestClassifier()
    model.fit(X,y)
    feat_importance = validation.get_feature_importance(model)
    assert (X.shape[1] == feat_importance.shape[0])
