from probatus.interpret.inspector import return_confusion_metric, InspectorShap, BaseInspector
from .mocks import MockClusterer, MockModel
from probatus.datasets import lending_club
from probatus.models import lending_club_model
from probatus.utils import NotFittedError
from unittest.mock import patch
from probatus.utils import NotFittedError, UnsupportedModelError

import numpy as np
import pandas as pd
import pytest

test_sensitivity = 0.0000000001

def test_return_confusion_metric():

    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)

    expected_output_not_normalized = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], dtype=float)
    expected_output_normalized = np.array([0.11111111, 0.22222222, 0.33333333, 0.22222222, 0.11111111, 0.], dtype=float)
    assert (expected_output_normalized - return_confusion_metric(y_true, y_score, normalize=True < test_sensitivity)).all()
    assert (expected_output_not_normalized - return_confusion_metric(y_true, y_score, normalize=False) < test_sensitivity).all()

@patch.object(MockClusterer, 'fit')
def test_fit_clusters__base_inspector(mock_clusterer):
    # Base Inspector case algotype is kmeans
    inspector = BaseInspector(algotype='kmeans')
    inspector.clusterer = mock_clusterer

    X = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])

    inspector.fit_clusters(X)
    # Check if has been called with correct argument
    inspector.clusterer.fit.assert_called_with(X)
    # Check if it has not been modified
    pd.testing.assert_frame_equal(X, X_copy)
    # Check if fitted flag has been changed correctly
    assert inspector.fitted is True


@patch.object(MockClusterer, 'fit')
def test_fit_clusters__inspector_shap(mock_clusterer):
    inspector = InspectorShap(model= MockModel(), algotype='kmeans', cluster_probability=False)
    inspector.clusterer = mock_clusterer

    X = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])

    assert inspector.cluster_probabilities is False
    assert inspector.predicted_proba is None

    # Check if the df has not been modified
    pd.testing.assert_frame_equal(X, X_copy)
    assert inspector.fitted is True
    #Check if not fitted exception is raised
    inspector.fit_clusters(X)
    # Check if X3 has not been modified
    pd.testing.assert_frame_equal(X, X_copy)

    inspector.clusterer.fit.assert_called_with(X)


@patch.object(MockClusterer, 'fit')
def test_fit_clusters__inspector_shap_proba(mock_clusterer):
    inspector = InspectorShap(model= MockModel(), algotype='kmeans', cluster_probability=True)
    inspector.clusterer = mock_clusterer
    inspector.predicted_proba = True

    X = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])

    assert inspector.fitted is False
    assert inspector.cluster_probabilities is True
    #Check if not fitted exception is raised
    inspector.fit_clusters(X)
    # Check if X3 has not been modified
    pd.testing.assert_frame_equal(X, X_copy)
    assert inspector.fitted is True


@patch.object(MockClusterer, 'predict')
def test_predict_clusters(mock_clusterer):
    mock_clusterer.predict.return_value = [1, 0]

    # Base Inspector case algotype is kmeans
    inspector1 = BaseInspector(algotype='kmeans')
    inspector1.clusterer = mock_clusterer

    # InspectorShap kmeans
    inspector2 = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=False)
    inspector2.clusterer = mock_clusterer

    # InspectorShap not fitted
    inspector3 = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=True)
    inspector3.clusterer = mock_clusterer
    inspector3.predicted_proba = [1, 0]

    X1 = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X1_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X2 = pd.DataFrame([[3, 2, 1], [1, 2, 3]])
    X2_copy = pd.DataFrame([[3, 2, 1], [1, 2, 3]])
    X3 = pd.DataFrame([[1, 1, 1], [0, 0, 0]])
    X3_copy = pd.DataFrame([[1, 1, 1], [0, 0, 0]])

    inspector1.fit_clusters(X1)
    # Check if the prediction is correct according to the Mock clusterer
    assert inspector1.predict_clusters(X1) == [1, 0]
    # Check if the clusterer was called with correct input
    inspector1.clusterer.predict.assert_called_with(X1)
    # Check if the X has not been modified
    pd.testing.assert_frame_equal(X1, X1_copy)

    inspector2.fit_clusters(X2)
    # Check if the output is correct, as should be according to MockClusterer
    assert inspector2.predict_clusters(X2) == [1, 0]
    # Check if the df has not been modified by the prediction
    pd.testing.assert_frame_equal(X2, X2_copy)

    # Check if not fitted exception is raised
    assert inspector3.fitted is False
    with pytest.raises(NotFittedError):
        inspector3.predict_clusters(X3)
    # Check if X3 has not been modified
    pd.testing.assert_frame_equal(X3, X3_copy)

def get_feats_and_model():
    _, X_train, y_train, X_test, y_test = lending_club()
    rf = lending_club_model()

    return rf, X_train, X_test, y_train, y_test

#@pytest.mark.skip(reason="Skip it for now for speed")
def test_inspector():
    rf, X_train, y_train, X_test, y_test = get_feats_and_model()

    test_inspector = InspectorShap(rf, n_clusters=4)
    test_inspector.inspect(X_train, y_train, approximate=False)

    # Check that the cluster numbers matches
    assert len(test_inspector.clusters.unique()) == 4

    report = test_inspector.get_report()

    assert report.shape == (4, 6)

    # TODO Fix the tests related to InspectorShap
    # expected_confusion = np.array([0.43190657, 0.06716497, 0.0319691, 0.18831297])
    # expected_confusion = np.array([0.21282713, 0.08869656, 0.56882355, 0.02859485])

    # The order might change - check the  sum of the values
    # assert (np.abs((report["average_confusion"].values - expected_confusion).sum()) < 0.05)

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

#@pytest.mark.skip(reason="Skip it for now for speed")
def test_inspector_with_eval_set():
    assert True

    rf, X_train, y_train, X_test, y_test = get_feats_and_model()

    test_inspector = InspectorShap(rf, n_clusters=4)

    # Make sure the assertion works if the samples names length does not match the eval set length
    with pytest.raises(AssertionError):
        test_inspector.inspect(X_train, y_train,
                               eval_set=[(X_train, y_train), (X_test, y_test)],
                               sample_names=['sample1'],
                               approximate=False)

    test_inspector.inspect(X_train, y_train,
                           eval_set=[(X_train, y_train), (X_test, y_test)],
                           sample_names=['sample1', 'samples'],
                           approximate=False)

    # dummy = test_inspector.get_report()

    real_train = test_inspector.slice_cluster(0)[0]
    eval_set_train = test_inspector.slice_cluster_eval_set(0)[0][0]

    assert real_train.equals(eval_set_train)

    assert len(test_inspector.slice_cluster_eval_set(0)) == 2
    assert len(test_inspector.slice_cluster_eval_set(0)[0]) == 3

    # assert that too if you look for high, returns an index error
    with pytest.raises(IndexError):
        test_inspector.slice_cluster_eval_set(0)[2]
