import numpy as np
import pandas as pd
import numbers

import pytest

from probatus.stat_tests import ks, psi, DistributionStatistics, AutoDist
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_missing_values_in_autodist():
    """Test missing values have no impact in AutoDist functionality. """
    # Create dummy dataframe
    X, y = make_classification(50, 5, random_state=0)
    X = pd.DataFrame(X)
    # Split train and test
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=1)
    # Define an add-on with only missing values
    X_na = pd.DataFrame(np.tile(np.nan,(X.shape[1], X.shape[1])))

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

def test_warnings_are_issued_for_missing():
    """Test if warnings are issued when missing values are present in the input of autodist."""
    # Generate a random input matrix
    X = pd.DataFrame({"A":[number for number in range (0, 50)]})
    X = X.assign(B = X['A'], C = X['A'], D = X['A'], E = X['A'])

    # Add some missing values to the dataframe.
    X_na = X.copy()
    X_na.iloc[X.sample(5, random_state=1).index,1:3] = np.nan

    # Test missing value removal on the first data input.
    with pytest.warns(None) as record_first:
        missing_first = AutoDist(statistical_tests=["PSI"], binning_strategies="SimpleBucketer", bin_count=10).compute(X_na, X)
    assert len(record_first) == 2

    # Test missing values removal on the second data input
    with pytest.warns(None) as record_second:
        missing_second = AutoDist(statistical_tests=["PSI"], binning_strategies="SimpleBucketer", bin_count=10).compute(X, X_na)
    assert len(record_second) == 2

    # Test the missing values removal on the first and second data input
    with pytest.warns(None) as record_both:
        missing_both = AutoDist(statistical_tests=["PSI"], binning_strategies="SimpleBucketer", bin_count=10).compute(X_na, X_na)
    assert len(record_both) == 2

    # Test case where there are no missing values
    with pytest.warns(None) as record_both:
        missing_both = AutoDist(statistical_tests=["PSI"], binning_strategies="SimpleBucketer", bin_count=10).compute(X, X)
    assert len(record_both) == 0
