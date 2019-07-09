import numpy as np
from pyrisk.stat_tests.ks import ks


def test_ks_returns_one():
    d1 = np.random.normal(size = 1000)
    d2 = d1
    assert ks(d1, d2) == 1.0


def test_psi_returns_small():
    d1 = np.random.normal(size = 1000)
    d2 = np.random.weibull(1, size = 1000) - 1
    assert ks(d1, d2) < 0.001