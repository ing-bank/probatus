import numpy as np
import numbers
from pyrisk.stat_tests import DistributionStatistics, ks, psi


def test_distribution_statistics_psi():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('psi', 'SimpleBucketer', bin_count=10)
    assert not myTest.fitted
    res = myTest.fit(d1, d2)
    assert myTest.fitted
    assert isinstance(res, numbers.Number)


def test_distribution_statistics_tuple_output():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('ks', 'SimpleBucketer', bin_count=10)
    assert not myTest.fitted
    res = myTest.fit(d1, d2)
    assert myTest.fitted
    assert isinstance(res, tuple)


def test_distribution_statistics_ks_no_binning():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('ks', binning_strategy=None)
    assert not myTest.fitted
    res = myTest.fit(d1, d2)
    assert myTest.fitted
    assert isinstance(res, tuple)


def test_distribution_statistics_attributes_psi():
    a = np.random.normal(size=1000)
    b = np.random.normal(size=1000)
    d1 = np.histogram(a, 10)[0]
    d2 = np.histogram(b, 10)[0]
    myTest = DistributionStatistics('psi', binning_strategy=None)
    _ = myTest.fit(d1, d2, verbose=False)
    psi_value = psi(d1, d2, verbose=False)
    assert myTest.statistic == psi_value


def test_distribution_statistics_attributes_ks():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.normal(size=1000), 10)[0]
    myTest = DistributionStatistics('ks', binning_strategy=None)
    _ = myTest.fit(d1, d2, verbose=False)
    ks_value, p_value = ks(d1, d2)
    assert myTest.statistic == ks_value
