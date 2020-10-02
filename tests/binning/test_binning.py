import numpy as np
import pytest

from probatus.binning import SimpleBucketer, QuantileBucketer, AgglomerativeBucketer, TreeBucketer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


@pytest.mark.filterwarnings("ignore:")
def test_deprecations():
    x = [1, 2, 1]
    bins = 3
    myBucketer = SimpleBucketer(bin_count=bins)
    myBucketer.fit(x)
    with pytest.deprecated_call():
        myBucketer.counts

    with pytest.deprecated_call():
        myBucketer.boundaries
    
    
def test_simple_bins():
    x = [1, 2, 1]
    bins = 3
    myBucketer = SimpleBucketer(bin_count=bins)
    with pytest.raises(NotFittedError):
       myBucketer.compute([1,2])
    
    myBucketer.fit(x)
    assert len(myBucketer.counts_) == bins
    assert np.array_equal(myBucketer.counts_, np.array([2, 0, 1]))
    assert len(myBucketer.boundaries_) == bins + 1
    np.testing.assert_array_almost_equal(myBucketer.boundaries_, np.array([1., 1.33333333, 1.66666667, 2.]))
    # test static method
    counts, boundaries = SimpleBucketer(bin_count=bins).simple_bins(x, bins)
    assert np.array_equal(myBucketer.counts_, counts)
    np.testing.assert_array_almost_equal(myBucketer.boundaries_, boundaries)
    assert repr(myBucketer).startswith('SimpleBucketer')


def test_quantile_bins():
    bins = 4
    random_state = np.random.RandomState(0)
    x = random_state.normal(0, 1, size=1000)
    myBucketer = QuantileBucketer(bin_count=bins)
    with pytest.raises(NotFittedError):
       myBucketer.compute([1,2])
    myBucketer.fit(x)
    assert len(myBucketer.counts_) == bins
    assert np.array_equal(myBucketer.counts_, np.array([250, 250, 250, 250]))
    assert len(myBucketer.boundaries_) == bins + 1
    np.testing.assert_array_almost_equal(myBucketer.boundaries_, np.array([-3.0, -0.7, -0.1, 0.6, 2.8]), decimal=1)
    # test static method
    counts, boundaries = QuantileBucketer(bin_count=bins).quantile_bins(x, bins)
    assert np.array_equal(myBucketer.counts_, counts)
    np.testing.assert_array_almost_equal(myBucketer.boundaries_, boundaries)
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
    with pytest.raises(NotFittedError):
       myBucketer.compute([1,2])
    myBucketer.fit(x)
    assert len(myBucketer.counts_) == bins
    assert np.array_equal(myBucketer.counts_, np.array([24, 16, 80, 80]))
    assert len(myBucketer.boundaries_) == bins + 1
    np.testing.assert_array_almost_equal(myBucketer.boundaries_, np.array([0, 0.11, 0.59, 0.88, 0.99]), decimal=2)
    # test static method
    counts, boundaries = AgglomerativeBucketer(bin_count=bins).agglomerative_clustering_binning(x, bins)
    assert np.array_equal(myBucketer.counts_, counts)
    np.testing.assert_array_almost_equal(myBucketer.boundaries_, boundaries)
    assert repr(myBucketer).startswith('AgglomerativeBucketer')


def test_compute():
    x = np.arange(10)
    bins = 5
    myBucketer = QuantileBucketer(bins)
    x_new = x
    with pytest.raises(NotFittedError):
        assert myBucketer.compute(x_new)
    myBucketer.fit(x)
    assert len(myBucketer.compute(x_new)) == bins
    np.testing.assert_array_equal(myBucketer.counts_, myBucketer.compute(x_new))
    np.testing.assert_array_equal(myBucketer.counts_, myBucketer.fit_compute(x_new))
    x_new = x + 100
    np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0]), myBucketer.compute(x_new))
    x_new = x - 100
    np.testing.assert_array_equal(np.array([10, 0, 0, 0, 0]), myBucketer.compute(x_new))
    x_new = [1, 1, 1, 4, 4, 7]
    np.testing.assert_array_equal(np.array([3, 0, 2, 1, 0]), myBucketer.compute(x_new))


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


def test_tree_bucketer():

    x = np.array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4,
       2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8, 5. ,
       5.2, 5.4, 5.6, 5.8, 6. , 6.2, 6.4, 6.6, 6.8, 7. , 7.2, 7.4, 7.6,
       7.8, 8. , 8.2, 8.4, 8.6, 8.8, 9. , 9.2, 9.4, 9.6, 9.8])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1])

    myTreeBucketer = TreeBucketer(inf_edges=False,max_depth=3,min_samples_leaf=10,random_state=42)

    with pytest.raises(NotFittedError):
       myTreeBucketer.compute([1,2])

    myTreeBucketer.fit(x,y)

    assert all(myTreeBucketer.counts_==np.array([21, 15, 14]))
    assert myTreeBucketer.bin_count==3
    assert all(myTreeBucketer.boundaries_ - np.array([0.0, 4.1, 7.1, 9.8])<0.01)

    # If infinite edges is False, it must get the edges of the x array
    assert myTreeBucketer.boundaries_[0] == 0
    assert myTreeBucketer.boundaries_[-1] == 9.8


    myTreeBucketer = TreeBucketer(inf_edges=True,max_depth=3, min_samples_leaf=10, random_state=42)

    myTreeBucketer.fit(x, y)
    # check that the infinite edges is True, then edges must be infinite
    assert myTreeBucketer.boundaries_[0] == -np.inf
    assert myTreeBucketer.boundaries_[-1] == +np.inf


def test_tree_bucketer_dependence():
    x = np.arange(0, 10, 0.01)
    y = [1 if z < 0.5 else 0 for z in np.random.uniform(size=x.shape[0])]


    # Test number of leaves is always within the expected ranges
    myTreeBucketer = TreeBucketer(inf_edges=False, max_depth=3, min_samples_leaf=10, random_state=42).fit(x,y)
    assert myTreeBucketer.bin_count<=np.power(2,myTreeBucketer.tree.max_depth)

    # Test number of leaves is always within the expected ranges
    myTreeBucketer = TreeBucketer(inf_edges=False, max_depth=6, min_samples_leaf=1, random_state=42).fit(x, y)
    assert myTreeBucketer.bin_count <= np.power(2, myTreeBucketer.tree.max_depth)

    # Test that the counts per bin never drop below min_samples_leaf
    myTreeBucketer = TreeBucketer(inf_edges=False, max_depth=6, min_samples_leaf=100, random_state=42).fit(x, y)
    assert all(myTreeBucketer.counts_ >= myTreeBucketer.tree.min_samples_leaf)

    myTreeBucketer = TreeBucketer(inf_edges=False, max_depth=6, min_samples_leaf=200, random_state=42).fit(x, y)
    assert all(myTreeBucketer.counts_ >= myTreeBucketer.tree.min_samples_leaf)

    # Test that if the leaf is set to the number of entries,it raises an Error
    myTreeBucketer = TreeBucketer(inf_edges=False, max_depth=6, min_samples_leaf=x.shape[0], random_state=42)

    with pytest.raises(ValueError):
        assert myTreeBucketer.fit(x, y)

    # Test that if the leaf is set to the number of entries-1, it returns only one bin
    myTreeBucketer = (
        TreeBucketer(inf_edges=False, max_depth=6, min_samples_leaf=x.shape[0]-1, random_state=42) \
        .fit(x, y)
    )
    assert myTreeBucketer.bin_count ==1
    assert all(myTreeBucketer.counts_ >= myTreeBucketer.tree.min_samples_leaf)


