import numpy as np
import pytest
from probatus.utils import NotFittedError

from probatus.binning import SimpleBucketer, QuantileBucketer, AgglomerativeBucketer


def test_simple_bins():
    x = [1, 2, 1]
    bins = 3
    myBucketer = SimpleBucketer(bin_count=bins)
    assert not myBucketer.fitted
    myBucketer.fit(x)
    assert myBucketer.fitted
    assert len(myBucketer.counts) == bins
    assert np.array_equal(myBucketer.counts, np.array([2, 0, 1]))
    assert len(myBucketer.boundaries) == bins + 1
    np.testing.assert_array_almost_equal(myBucketer.boundaries, np.array([1., 1.33333333, 1.66666667, 2.]))
    # test static method
    counts, boundaries = SimpleBucketer(bin_count=bins).simple_bins(x, bins)
    assert np.array_equal(myBucketer.counts, counts)
    np.testing.assert_array_almost_equal(myBucketer.boundaries, boundaries)
    assert repr(myBucketer).startswith('SimpleBucketer')


def test_quantile_bins():
    bins = 4
    random_state = np.random.RandomState(0)
    x = random_state.normal(0, 1, size=1000)
    myBucketer = QuantileBucketer(bin_count=bins)
    assert not myBucketer.fitted
    myBucketer.fit(x)
    assert myBucketer.fitted
    assert len(myBucketer.counts) == bins
    assert np.array_equal(myBucketer.counts, np.array([250, 250, 250, 250]))
    assert len(myBucketer.boundaries) == bins + 1
    np.testing.assert_array_almost_equal(myBucketer.boundaries, np.array([-3.0, -0.7, -0.1, 0.6, 2.8]), decimal=1)
    # test static method
    counts, boundaries = QuantileBucketer(bin_count=bins).quantile_bins(x, bins)
    assert np.array_equal(myBucketer.counts, counts)
    np.testing.assert_array_almost_equal(myBucketer.boundaries, boundaries)
    # test inf edges
    counts, boundaries = QuantileBucketer(bin_count=bins).quantile_bins(x, bins, inf_edges=True)
    assert boundaries[0] == -np.inf
    assert boundaries[-1] == np.inf
    assert repr(myBucketer).startswith('QuantileBucketer')


def test_agglomerative_clustering_new():
    def log_function(x):
        return 1 / (1 + np.exp(-10 * x))

    x = [log_function(x) for x in np.arange(-1, 1, 0.01)]
    bins = 4
    myBucketer = AgglomerativeBucketer(bin_count=bins)
    assert not myBucketer.fitted
    myBucketer.fit(x)
    assert myBucketer.fitted
    assert len(myBucketer.counts) == bins
    assert np.array_equal(myBucketer.counts, np.array([24, 16, 80, 80]))
    assert len(myBucketer.boundaries) == bins + 1
    np.testing.assert_array_almost_equal(myBucketer.boundaries, np.array([0, 0.11, 0.59, 0.88, 0.99]), decimal=2)
    # test static method
    counts, boundaries = AgglomerativeBucketer(bin_count=bins).agglomerative_clustering_binning(x, bins)
    assert np.array_equal(myBucketer.counts, counts)
    np.testing.assert_array_almost_equal(myBucketer.boundaries, boundaries)
    assert repr(myBucketer).startswith('AgglomerativeBucketer')


def test_apply_bucketing():
    x = np.arange(10)
    bins = 5
    myBucketer = QuantileBucketer(bins)
    x_new = x
    with pytest.raises(NotFittedError):
        assert myBucketer.apply_bucketing(x_new)
    myBucketer.fit(x)
    assert len(myBucketer.apply_bucketing(x_new)) == bins
    np.testing.assert_array_equal(myBucketer.counts, myBucketer.apply_bucketing(x_new))
    x_new = x + 100
    np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0]), myBucketer.apply_bucketing(x_new))
    x_new = x - 100
    np.testing.assert_array_equal(np.array([10, 0, 0, 0, 0]), myBucketer.apply_bucketing(x_new))
    x_new = [1, 1, 1, 4, 4, 7]
    np.testing.assert_array_equal(np.array([3, 0, 2, 1, 0]), myBucketer.apply_bucketing(x_new))


def test_quantile_with_unique_values():
    np.random.seed(42)
    dist_0_1 = np.random.uniform(size=20)
    dist_peak_at_0 = np.zeros(shape=20)

    skewed_dist = np.hstack((dist_0_1, dist_peak_at_0))
    actual_out = QuantileBucketer(10).quantile_bins(skewed_dist, 10)

    expected_out = (
        np.array([20, 4, 4, 4, 4, 4]),
        np.array([0., 0.01894458, 0.23632033, 0.42214475, 0.60977678, 0.67440958, 0.99940487])
    )

    assert (actual_out[0] == expected_out[0]).all()
