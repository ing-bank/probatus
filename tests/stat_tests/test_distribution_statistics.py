import numpy as np
import pandas as pd
import numbers

import pytest

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from probatus.stat_tests import ks, psi, DistributionStatistics, AutoDist


def test_distribution_statistics_base():
    """
    Test.
    """
    with pytest.raises(NotImplementedError):
        assert DistributionStatistics("doesnotexist", "SimpleBucketer", bin_count=10)
    with pytest.raises(NotImplementedError):
        assert DistributionStatistics("psi", "doesnotexist", bin_count=10)
    myTest = DistributionStatistics("psi", "SimpleBucketer", bin_count=10)
    assert repr(myTest).startswith("DistributionStatistics")


def test_distribution_statistics_psi():
    """
    Test.
    """
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics("psi", "SimpleBucketer", bin_count=10)
    assert not myTest.fitted
    psi_test, p_value_test = myTest.compute(d1, d2)
    assert myTest.fitted
    assert isinstance(psi_test, numbers.Number)


def test_distribution_statistics_tuple_output():
    """
    Test.
    """
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics("ks", "SimpleBucketer", bin_count=10)
    assert not myTest.fitted
    res = myTest.compute(d1, d2)
    assert myTest.fitted
    assert isinstance(res, tuple)


def test_distribution_statistics_ks_no_binning():
    """
    Test.
    """
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics("ks", binning_strategy=None)
    assert not myTest.fitted
    res = myTest.compute(d1, d2)
    assert myTest.fitted
    assert isinstance(res, tuple)


def test_distribution_statistics_attributes_psi():
    """
    Test.
    """
    a = np.random.normal(size=1000)
    b = np.random.normal(size=1000)
    d1 = np.histogram(a, 10)[0]
    d2 = np.histogram(b, 10)[0]
    myTest = DistributionStatistics("psi", binning_strategy=None)
    _ = myTest.compute(d1, d2, verbose=False)
    psi_value_test, p_value_test = psi(d1, d2, verbose=False)
    assert myTest.statistic == psi_value_test


def test_distribution_statistics_attributes_ks():
    """
    Test.
    """
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.normal(size=1000), 10)[0]
    myTest = DistributionStatistics("ks", binning_strategy=None)
    _ = myTest.compute(d1, d2, verbose=False)
    ks_value, p_value = ks(d1, d2)
    assert myTest.statistic == ks_value


def test_distribution_statistics_autodist_base():
    """
    Test.
    """
    nr_features = 2
    size = 1000
    np.random.seed(0)
    df1 = pd.DataFrame(np.random.normal(size=(size, nr_features)), columns=[f"feat_{x}" for x in range(nr_features)])
    df2 = pd.DataFrame(np.random.normal(size=(size, nr_features)), columns=[f"feat_{x}" for x in range(nr_features)])
    features = df1.columns
    myAutoDist = AutoDist(statistical_tests="all", binning_strategies="all", bin_count=[10, 20])
    assert repr(myAutoDist).startswith("AutoDist")
    assert not myAutoDist.fitted
    res = myAutoDist.compute(df1, df2, column_names=features)
    assert myAutoDist.fitted
    pd.testing.assert_frame_equal(res, myAutoDist.result)
    assert isinstance(res, pd.DataFrame)
    assert res["column"].values.tolist() == features.to_list()

    dist = DistributionStatistics(statistical_test="ks", binning_strategy="simplebucketer", bin_count=10)
    dist.compute(df1["feat_0"], df2["feat_0"])
    assert dist.p_value == res.loc[res["column"] == "feat_0", "p_value_KS_simplebucketer_10"][0]
    assert dist.statistic == res.loc[res["column"] == "feat_0", "statistic_KS_simplebucketer_10"][0]

    dist = DistributionStatistics(statistical_test="ks", binning_strategy=None, bin_count=10)
    dist.compute(df1["feat_0"], df2["feat_0"])
    assert dist.p_value == res.loc[res["column"] == "feat_0", "p_value_KS_no_bucketing_0"][0]
    assert dist.statistic == res.loc[res["column"] == "feat_0", "statistic_KS_no_bucketing_0"][0]


def test_distribution_statistics_autodist_column_names_error():
    """
    Test.
    """
    df1 = pd.DataFrame({"feat_0": [1, 2, 3, 4, 5], "feat_1": [5, 6, 7, 8, 9]})
    df2 = df1
    features = df1.columns.values.tolist() + ["missing_feature"]
    myAutoDist = AutoDist()
    with pytest.raises(Exception):
        assert myAutoDist.compute(df1, df2, column_names=features)

    df1 = pd.DataFrame({"feat_0": [1, 2, 3, 4, 5], "feat_1": [5, 6, 7, 8, 9]})
    df2 = df1.copy()
    df1["feat_2"] = 0
    features = df2.columns.values.tolist() + ["missing_feature"]
    myAutoDist = AutoDist()
    with pytest.raises(Exception):
        assert myAutoDist.compute(df1, df2, column_names=features)


def test_distribution_statistics_autodist_return_failed_tests():
    """
    Test.
    """
    df1 = pd.DataFrame({"feat_0": [1, 2, 3, 4, 5], "feat_1": [5, 6, 7, 8, 9]})
    df2 = df1
    features = df1.columns.values.tolist()
    myAutoDist = AutoDist(binning_strategies="all")
    res = myAutoDist.compute(df1, df2, column_names=features, return_failed_tests=True)
    assert res.isin(["an error occurred"]).any().any()
    res = myAutoDist.compute(df1, df2, column_names=features, return_failed_tests=False)
    assert not res.isin(["an error occurred"]).any().any()


def test_distribution_statistics_autodist_default():
    """
    Test.
    """
    df1 = pd.DataFrame({"feat_0": [1, 2, 3, 4, 5], "feat_1": [5, 6, 7, 8, 9]})
    df2 = df1
    features = df1.columns.values.tolist()
    myAutoDist = AutoDist(binning_strategies="default", bin_count=10)
    res = myAutoDist.compute(df1, df2, column_names=features)
    for stat_test, stat_info in DistributionStatistics.statistical_test_dict.items():
        if stat_info["default_binning"]:
            assert f"p_value_{stat_test}_{stat_info['default_binning']}_10" in res.columns
        else:
            assert f"p_value_{stat_test}_no_bucketing_0" in res.columns

    assert "p_value_agglomerativebucketer_10" not in res.columns
    assert res.shape == (len(df1.columns), 1 + 2 * len(DistributionStatistics.statistical_test_dict))


def test_distribution_statistics_autodist_init():
    """
    Test.
    """
    myAutoDist = AutoDist(statistical_tests="all", binning_strategies="all")
    assert isinstance(myAutoDist.statistical_tests, list)
    myAutoDist = AutoDist(statistical_tests="ks", binning_strategies="all")
    assert myAutoDist.statistical_tests == ["ks"]
    myAutoDist = AutoDist(statistical_tests=["ks", "psi"], binning_strategies="all")
    assert myAutoDist.statistical_tests == ["ks", "psi"]

    myAutoDist = AutoDist(statistical_tests="all", binning_strategies="all")
    assert isinstance(myAutoDist.binning_strategies, list)
    myAutoDist = AutoDist(statistical_tests="all", binning_strategies="quantilebucketer")
    assert myAutoDist.binning_strategies == ["quantilebucketer"]
    myAutoDist = AutoDist(statistical_tests="all", binning_strategies=["quantilebucketer", "simplebucketer"])
    assert myAutoDist.binning_strategies == ["quantilebucketer", "simplebucketer"]


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
    # Generate an input dataframe without missing values
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
