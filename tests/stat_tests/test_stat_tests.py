import numpy as np
import pandas as pd
from probatus.stat_tests import ks, psi, es, ad, sw

from probatus.binning import binning



def test_psi_returns_zero():
    '''Population Stability Index returns zero'''
    x = np.random.normal(size=1000)
    myBucketer = binning.QuantileBucketer(bin_count=10)
    myBucketer.fit(x)
    d1 = myBucketer.counts_
    d2 = d1
    psi_test, p_value_test = psi(d1, d2, verbose=False)
    assert psi_test == 0.0


def test_psi_returns_large():
    '''Population Stability Index returns large value (>1.0)'''
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    psi_test, p_value_test = psi(d1, d2, verbose=False)
    assert psi_test > 1.0


def test_ks_returns_one():
    '''Kolmogorov-Smirnov test statistic returns one'''
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert ks(d1, d2)[1] == 1.0


def test_ks_accepts_pd_series():
    '''Kolmogorov-Smirnov test statistic accepts pd Series'''
    d1 = pd.Series(np.random.normal(size=1000))
    d2 = d1
    assert ks(d1, d2)[1] == 1.0


def test_ks_returns_small():
    '''Kolmogorov-Smirnov test statistic returns small value (<0.001)'''
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert ks(d1, d2)[1] < 0.001


def test_es_returns_one():
    '''Epps-Singleton test statistic returns one'''
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert es(d1, d2)[1] == 1.0


def test_es_returns_small():
    '''Epps-Singleton test statistic returns small value (<0.001)'''
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert es(d1, d2)[1] < 0.001


def test_ad_returns_big():
    '''Anderson-Darling test statistic returns big value (>=0.25)'''
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert ad(d1, d2)[1] >= 0.25


def test_ad_returns_small():
    '''Anderson-Darling test statistic returns small value (<=0.001)'''
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert ad(d1, d2)[1] <= 0.001


def test_sw_returns_zero():
    '''Shapiro-Wilk test statistic returns zero'''
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert sw(d1, d2)[0] == 0
    