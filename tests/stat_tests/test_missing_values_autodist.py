import numpy as np
import pandas as pd
import numbers

import pytest

from probatus.stat_tests import ks, psi, DistributionStatistics, AutoDist
from sklearn.datasets import make_classification

def test_missing_values_in_autodist():
    """Test missing values have no impact in AutoDist functionality. """
    # Create dummy dataframe
    X, y = make_classification(10000, 10)
    X = pd.DataFrame(X)
    # Split train and test
    X_train = X.sample(8000)
    X_test = X.sample(2000)
    # Define an add-on with only missing values
    X_na = pd.DataFrame(np.tile(np.nan,(10, 10)))

    # Compute the statistics with the missing values
    with_missings = (
        AutoDist(statistical_tests=["PSI", "KS"], binning_strategies="SimpleBucketer", bin_count=10)
        .compute(pd.concat([X_train, X_na]), pd.concat([X_test, X_na])))
        
    # Compute the statistics withpout the missing values
    no_missing = (
        AutoDist(statistical_tests=["PSI","KS"], binning_strategies="SimpleBucketer", bin_count=10)
        .compute(X_train, X_test))
        
    # Test the two set of results are identical
    pd.testing.assert_frame_equal(with_missings, no_missing)


