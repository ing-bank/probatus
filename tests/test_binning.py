import numpy as np
from pyrisk.binning import SimpleBucketer, QuantileBucketer, AgglomerativeBucketer


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
