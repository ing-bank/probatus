from sklearn.datasets import make_classification
from probatus.samples import ResemblanceModel
from unittest.mock import patch
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


def test_compute():
    rm = ResemblanceModel(model_type='lr')

    importances = pd.Series([0.5, 0.3, 0.6, 0.01, 0.05], index=['f1', 'f2', 'f3', 'f4', 'f5'])

    expected_outcome = pd.Series([0.6, 0.5, 0.3, 0.05], index=['f3', 'f1', 'f2', 'f5'])

    with patch.object(ResemblanceModel, '_get_feature_importance') as mock_get_feature_importance:
        mock_get_feature_importance.return_value = importances

        output = rm.compute(sort=True, threshold=0.05)
    pd.testing.assert_series_equal(output, expected_outcome)


