import numpy as np
import numbers
from pyrisk.stat_tests import DistributionStatistics


def test_distribution_statistics_psi():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    myTest = DistributionStatistics('psi', 'SimpleBucketer', bin_count=10)
    assert not myTest.fitted
    # note that `n` and `m` only need to be specified for PSI
    res = myTest.fit(d1, d2, n=len(d1), m=len(d2))
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
