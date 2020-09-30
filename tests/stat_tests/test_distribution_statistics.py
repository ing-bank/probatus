import numpy as np
import pandas as pd
import numbers

import pytest

from probatus.stat_tests import ks, psi, DistributionStatistics, AutoDist


def test_distribution_statistics_base():
    with pytest.raises(NotImplementedError):
        assert DistributionStatistics('doesnotexist', 'SimpleBucketer', bin_count=10)
    with pytest.raises(NotImplementedError):
        assert DistributionStatistics('psi', 'doesnotexist', bin_count=10)
    myTest = DistributionStatistics('psi', 'SimpleBucketer', bin_count=10)
    assert repr(myTest).startswith('DistributionStatistics')


def test_distribution_statistics_psi():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('psi', 'SimpleBucketer', bin_count=10)
    assert not myTest.fitted
    psi_test, p_value_test = myTest.compute(d1, d2)
    assert myTest.fitted
    assert isinstance(psi_test, numbers.Number)


def test_distribution_statistics_tuple_output():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('ks', 'SimpleBucketer', bin_count=10)
    assert not myTest.fitted
    res = myTest.compute(d1, d2)
    assert myTest.fitted
    assert isinstance(res, tuple)


def test_distribution_statistics_ks_no_binning():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('ks', binning_strategy=None)
    assert not myTest.fitted
    res = myTest.compute(d1, d2)
    assert myTest.fitted
    assert isinstance(res, tuple)


def test_distribution_statistics_attributes_psi():
    a = np.random.normal(size=1000)
    b = np.random.normal(size=1000)
    d1 = np.histogram(a, 10)[0]
    d2 = np.histogram(b, 10)[0]
    myTest = DistributionStatistics('psi', binning_strategy=None)
    _ = myTest.compute(d1, d2, verbose=False)
    psi_value_test, p_value_test = psi(d1, d2, verbose=False)
    assert myTest.statistic == psi_value_test


def test_distribution_statistics_attributes_ks():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.normal(size=1000), 10)[0]
    myTest = DistributionStatistics('ks', binning_strategy=None)
    _ = myTest.compute(d1, d2, verbose=False)
    ks_value, p_value = ks(d1, d2)
    assert myTest.statistic == ks_value


def test_distribution_statistics_autodist_base():
    nr_features = 2
    size = 1000
    np.random.seed(0)
    df1 = pd.DataFrame(np.random.normal(size=(size, nr_features)), columns=[f'feat_{x}' for x in range(nr_features)])
    df2 = pd.DataFrame(np.random.normal(size=(size, nr_features)), columns=[f'feat_{x}' for x in range(nr_features)])
    features = df1.columns
    myAutoDist = AutoDist(statistical_tests='all', binning_strategies='all', bin_count=[10, 20])
    assert repr(myAutoDist).startswith('AutoDist')
    assert not myAutoDist.fitted
    res = myAutoDist.compute(df1, df2, column_names=features)
    assert myAutoDist.fitted
    pd.testing.assert_frame_equal(res, myAutoDist.result)
    assert isinstance(res, pd.DataFrame)
    assert res['column'].values.tolist() == features.to_list()

    dist = DistributionStatistics(statistical_test='ks', binning_strategy='simplebucketer', bin_count=10)
    dist.compute(df1['feat_0'], df2['feat_0'])
    assert dist.p_value == res.loc[res['column'] == 'feat_0', 'p_value_KS_simplebucketer_10'][0]
    assert dist.statistic == res.loc[res['column'] == 'feat_0', 'statistic_KS_simplebucketer_10'][0]

    dist = DistributionStatistics(statistical_test='ks', binning_strategy=None, bin_count=10)
    dist.compute(df1['feat_0'], df2['feat_0'])
    assert dist.p_value == res.loc[res['column'] == 'feat_0', 'p_value_KS_no_bucketing_0'][0]
    assert dist.statistic == res.loc[res['column'] == 'feat_0', 'statistic_KS_no_bucketing_0'][0]


def test_distribution_statistics_autodist_column_names_error():
    df1 = pd.DataFrame({'feat_0': [1, 2, 3, 4, 5], 'feat_1': [5, 6, 7, 8, 9]})
    df2 = df1
    features = df1.columns.values.tolist() + ['missing_feature']
    myAutoDist = AutoDist()
    with pytest.raises(Exception):
        assert myAutoDist.compute(df1, df2, column_names=features)

    df1 = pd.DataFrame({'feat_0': [1, 2, 3, 4, 5], 'feat_1': [5, 6, 7, 8, 9]})
    df2 = df1.copy()
    df1['feat_2'] = 0
    features = df2.columns.values.tolist() + ['missing_feature']
    myAutoDist = AutoDist()
    with pytest.raises(Exception):
        assert myAutoDist.compute(df1, df2, column_names=features)


def test_distribution_statistics_autodist_return_failed_tests():
    df1 = pd.DataFrame({'feat_0': [1, 2, 3, 4, 5], 'feat_1': [5, 6, 7, 8, 9]})
    df2 = df1
    features = df1.columns.values.tolist()
    myAutoDist = AutoDist(binning_strategies="all")
    res = myAutoDist.compute(df1, df2, column_names=features, return_failed_tests=True)
    assert res.isin(['an error occurred']).any().any()
    res = myAutoDist.compute(df1, df2, column_names=features, return_failed_tests=False)
    assert not res.isin(['an error occurred']).any().any()
    
def test_distribution_statistics_autodist_default():
    df1 = pd.DataFrame({'feat_0': [1, 2, 3, 4, 5], 'feat_1': [5, 6, 7, 8, 9]})
    df2 = df1
    features = df1.columns.values.tolist()
    myAutoDist = AutoDist(binning_strategies="default", bin_count=10)
    res = myAutoDist.compute(df1, df2, column_names=features)
    for stat_test, stat_info in DistributionStatistics.statistical_test_dict.items():
        if stat_info['default_binning']:
            assert f"p_value_{stat_test}_{stat_info['default_binning']}_10" in res.columns
        else:
            assert f"p_value_{stat_test}_no_bucketing_0" in res.columns

    assert "p_value_agglomerativebucketer_10" not in res.columns
    assert res.shape == (len(df1.columns), 1 + 2 * len(DistributionStatistics.statistical_test_dict))

def test_distribution_statistics_autodist_init():
    myAutoDist = AutoDist(statistical_tests='all', binning_strategies='all')
    assert isinstance(myAutoDist.statistical_tests, list)
    myAutoDist = AutoDist(statistical_tests='ks', binning_strategies='all')
    assert myAutoDist.statistical_tests == ['ks']
    myAutoDist = AutoDist(statistical_tests=['ks', 'psi'], binning_strategies='all')
    assert myAutoDist.statistical_tests == ['ks', 'psi']

    myAutoDist = AutoDist(statistical_tests='all', binning_strategies='all')
    assert isinstance(myAutoDist.binning_strategies, list)
    myAutoDist = AutoDist(statistical_tests='all', binning_strategies='quantilebucketer')
    assert myAutoDist.binning_strategies == ['quantilebucketer']
    myAutoDist = AutoDist(statistical_tests='all', binning_strategies=['quantilebucketer', 'simplebucketer'])
    assert myAutoDist.binning_strategies == ['quantilebucketer', 'simplebucketer']
