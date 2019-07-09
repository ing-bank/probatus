import numpy as np
from pyrisk.stat_tests.psi import psi 


def test_psi_returns_zero():
    d1 = np.random.normal(size = 1000)
    d2 = d1
    assert psi(d1, d2) == 0.0


def test_psi_returns_large():
    d1 = np.random.normal(size = 1000)
    d2 = np.random.weibull(1, size = 1000) - 1
    assert psi(d1, d2) > 1.0
