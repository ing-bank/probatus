import numpy as np
import pandas as pd
from probatus.stat_tests import ks, psi, es, ad, sw

from probatus.binning import binning


def test_psi_returns_zero():
    """
    Test.
    """
    x = np.random.normal(size=1000)
    myBucketer = binning.QuantileBucketer(bin_count=10)
    myBucketer.fit(x)
    d1 = myBucketer.counts_
    d2 = d1
    psi_test, p_value_test = psi(d1, d2, verbose=False)
    assert psi_test == 0.0


def test_psi_returns_large():
    """
    Test.
    """
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    psi_test, p_value_test = psi(d1, d2, verbose=False)
    assert psi_test > 1.0


def test_ks_returns_one():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert ks(d1, d2)[1] == 1.0


def test_ks_accepts_pd_series():
    """
    Test.
    """
    d1 = pd.Series(np.random.normal(size=1000))
    d2 = d1
    assert ks(d1, d2)[1] == 1.0


def test_ks_returns_small():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert ks(d1, d2)[1] < 0.001


def test_es_returns_one():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert es(d1, d2)[1] == 1.0


def test_es_returns_small():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert es(d1, d2)[1] < 0.001


def test_ad_returns_big():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert ad(d1, d2)[1] >= 0.25


def test_ad_returns_small():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert ad(d1, d2)[1] <= 0.001


def test_sw_returns_zero():
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert sw(d1, d2)[0] == 0
