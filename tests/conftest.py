import pytest
from unittest.mock import Mock
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="function")
def mock_model():
    """
    Fixture.
    """
    return Mock()


@pytest.fixture(scope="function")
def complex_data():
    """
    Fixture.
    """

    feature_names = ["f1_categorical", "f2_missing", "f3_static", "f4", "f5"]

    # Prepare two samples
    X, y = make_classification(
        n_samples=50,
        class_sep=0.05,
        n_informative=2,
        n_features=5,
        random_state=0,
        n_redundant=2,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X, columns=feature_names)
    X["f1_categorical"] = X["f1_categorical"].astype("category")
    X.loc[0:10, "f2_missing"] = np.nan
    return X, y


@pytest.fixture(scope="function")
def complex_data_split(complex_data):
    """
    Fixture.
    """
    X, y = complex_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="function")
def complex_lightgbm():
    """
    Fixture.
    """
    import lightgbm

    return lightgbm.LGBMClassifier(max_depth=5, num_leaves=11, class_weight="balanced", random_state=42)


@pytest.fixture(scope="function")
def complex_fitted_lightgbm(complex_data_split, complex_lightgbm):
    """
    Fixture.
    """
    X_train, _, y_train, _ = complex_data_split
    X_train["f1_categorical"] = X_train["f1_categorical"].astype("category")

    return complex_lightgbm.fit(X_train, y_train)
