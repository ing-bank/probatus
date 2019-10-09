from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from probatus.samples import ResemblanceModel
import numpy as np
import pandas as pd



def test_resemblance_random_split():
    X, dummy = make_classification(n_samples=10000,
                               n_classes=2,
                               random_state=1)

    X = pd.DataFrame(X)
    X1 = X.sample(frac=0.7)
    X2= X.loc[[ix for ix in X.index if ix not in X1.index]]


    rm = ResemblanceModel(model_type='rf').fit(X1,X2)

    assert np.abs(rm.auc_test-0.5<0.02)

def test_resemblance_random_split_with_lr():
    X, dummy = make_classification(n_samples=10000,
                                   n_classes=2,
                                   random_state=1)

    X = pd.DataFrame(X)
    X1 = X.sample(frac=0.7)
    X2 = X.loc[[ix for ix in X.index if ix not in X1.index]]

    rm = ResemblanceModel(model_type='lr').fit(X1, X2, random_state=42)

    assert np.abs(rm.auc_test - 0.5 < 0.02)




# def test_feature_importance():
#
#     X, y = make_classification(n_samples=10000, n_classes=2, random_state=1)
#     model = RandomForestClassifier()
#     model.fit(X,y)
#     feat_importance = validation.get_feature_importance(model)
#     assert (X.shape[1] == feat_importance.shape[0])
