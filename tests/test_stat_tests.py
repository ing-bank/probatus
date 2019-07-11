import numpy as np
import pandas as pd

from pyrisk.stat_tests import es, ks, psi


def test_psi_returns_zero():
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert psi(d1, d2) == 0.0


def test_psi_returns_large():
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.weibull(1, size=1000) - 1, 10)[0]
    assert psi(d1, d2) > 1.0


def test_ks_returns_one():
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert ks(d1, d2)[1] == 1.0


def test_ks_accepts_pd_series():
    d1 = pd.Series(np.random.normal(size=1000))
    d2 = d1
    assert ks(d1, d2)[1] == 1.0


def test_ks_returns_small():
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert ks(d1, d2)[1] < 0.001


def test_es_returns_one():
    d1 = np.random.normal(size=1000)
    d2 = d1
    assert es(d1, d2)[1] == 1.0


def test_es_returns_small():
    d1 = np.random.normal(size=1000)
    d2 = np.random.weibull(1, size=1000) - 1
    assert es(d1, d2)[1] < 0.001
