from probatus.interpret import InspectorShap
from probatus.datasets import lending_club
from probatus.models import lending_club_model
from probatus.utils import NotFittedError

import numpy as np
import pytest


def get_feats_and_model():
    _, X_train, y_train, X_test, y_test = lending_club()
    rf = lending_club_model()

    return rf, X_train, X_test, y_train, y_test



def test_inspector():
    rf, X_train, y_train, X_test, y_test = get_feats_and_model()

    test_inspector = InspectorShap(rf, n_clusters=4)
    test_inspector.inspect(X_train, y_train, approximate=True)

    # Check that the cluster numbers matches
    assert len(test_inspector.clusters.unique()) == 4

    report = test_inspector.get_report()

    assert report.shape == (4, 6)

    expected_confusion = np.array([0.43190657, 0.06716497, 0.0319691, 0.18831297])

    # The order might change - check the  sum of the values
    assert (np.abs((report["average_confusion"].values.sum() - expected_confusion.sum())) < 0.05)

    # Test slicing
    clust_slice = test_inspector.slice_cluster(3)
    compl_clust_slice = test_inspector.slice_cluster(3, complementary=True)

    assert len(clust_slice) == 3
    assert len(compl_clust_slice) == 3

    # Check thqat there is no index overlap between complementary slices
    assert len(set(clust_slice[0].index).intersection(compl_clust_slice[0].index)) == 0

    # check that slicing the cluster of the eval set raises  an exception
    with pytest.raises(NotFittedError):
        assert test_inspector.slice_cluster_eval_set(3)


def test_inspector_with_eval_set():
    assert True

    rf, X_train, y_train, X_test, y_test = get_feats_and_model()

    test_inspector = InspectorShap(rf, n_clusters=4)

    # Make sure the assertion works if the samples names length does not match the eval set length
    with pytest.raises(AssertionError):
        test_inspector.inspect(X_train, y_train,
                               eval_set=[(X_train, y_train), (X_test, y_test)],
                               sample_names=['sample1'],
                               approximate=True)

    test_inspector.inspect(X_train, y_train,
                           eval_set=[(X_train, y_train), (X_test, y_test)],
                           sample_names=['sample1', 'samples'],
                           approximate=True)

    # dummy = test_inspector.get_report()

    real_train = test_inspector.slice_cluster(0)[0]
    eval_set_train = test_inspector.slice_cluster_eval_set(0)[0][0]

    assert real_train.equals(eval_set_train)

    assert len(test_inspector.slice_cluster_eval_set(0)) == 2
    assert len(test_inspector.slice_cluster_eval_set(0)[0]) == 3

    # assert that too if you look for high, returns an index error
    with pytest.raises(IndexError):
        test_inspector.slice_cluster_eval_set(0)[2]
