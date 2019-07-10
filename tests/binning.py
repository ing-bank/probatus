import numpy as np
import pytest

from pyrisk.binning.binning import simple_bins, agglomerative_clustering_binning


def test_simple_bins():
    x = [1, 2, 1]
    bins = 3
    res = simple_bins(x, bins)
    assert (np.array_equal(res[0], np.array([2, 0, 1]))) and (
        np.array_equal(np.round(res[1]), np.round(np.array([1., 1.33333333, 1.66666667, 2.]))))


def test_agglomerative_clustering():
    def log_function(x):
        return 1 / (1 + np.exp(-10 * x))

    x = [log_function(x) for x in np.arange(-1, 1, 0.01)]
    bins = 4
    counts, boundaries = agglomerative_clustering_binning(x, bins)
    assert sum(counts) == len(x)
    assert boundaries[0] == min(x)
    assert boundaries[-1] == max(x)
    assert boundaries[1] == pytest.approx(0.11, abs=0.1)
    assert boundaries[2] == pytest.approx(0.58, abs=0.1)
    assert boundaries[3] == pytest.approx(0.87, abs=0.1)
