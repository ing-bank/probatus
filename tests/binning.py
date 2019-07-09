import numpy as np
from pyrisk.binning import simple_bins

def test_simple_bins():
    x = [1, 2, 1]
    bins = 3
    res = simple_bins(x,bins)
    assert  (np.array_equal(res[0],np.array([2, 0, 1]))) and (np.array_equal(np.round(res[1]), np.round(np.array([1., 1.33333333, 1.66666667, 2.]))))