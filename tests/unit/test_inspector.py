from probatus.interpret.inspector import return_confusion_metric, InspectorShap, BaseInspector
from .mocks import MockClusterer, MockModel
from probatus.datasets import lending_club
from probatus.models import lending_club_model
from unittest.mock import patch
from probatus.utils import NotFittedError, UnsupportedModelError

import numpy as np
import pandas as pd
import pytest

test_sensitivity = 0.0000000001


def test_return_confusion_metric__array():

    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)

    expected_output_not_normalized = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], dtype=float)
    expected_output_normalized = np.array([0.11111111, 0.22222222, 0.33333333, 0.22222222, 0.11111111, 0.], dtype=float)
    assert (expected_output_normalized - return_confusion_metric(y_true, y_score, normalize=True < test_sensitivity)).all()
    assert (expected_output_not_normalized - return_confusion_metric(y_true, y_score, normalize=False) < test_sensitivity).all()


def test_return_confusion_metric__series():
    # The method also needs to work with series, since it is called with series by create summary df
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_score = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    expected_output_not_normalized = pd.Series([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], dtype=float)
    expected_output_normalized = pd.Series([0.11111111, 0.22222222, 0.33333333, 0.22222222, 0.11111111, 0.], dtype=float)
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
    #Check if not fitted exception is raised
    inspector.fit_clusters(X)
    # Check if X3 has not been modified
    pd.testing.assert_frame_equal(X, X_copy)
    assert inspector.fitted is True


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
def test_predict_clusters__base_inspector(mock_clusterer):
    mock_clusterer.predict.return_value = [1, 0]

    inspector = BaseInspector(algotype='kmeans')
    inspector.clusterer = mock_clusterer

    X = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])

    inspector.fit_clusters(X)
    # Check if the prediction is correct according to the Mock clusterer
    assert inspector.predict_clusters(X) == [1, 0]
    # Check if the clusterer was called with correct input
    inspector.clusterer.predict.assert_called_with(X)
    # Check if the X has not been modified
    pd.testing.assert_frame_equal(X, X_copy)


@patch.object(MockClusterer, 'predict')
def test_predict_clusters__inspector_shap(mock_clusterer):
    mock_clusterer.predict.return_value = [1, 0]

    inspector = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=False)
    inspector.clusterer = mock_clusterer

    X = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])

    inspector.fit_clusters(X)
    # Check if the output is correct, as should be according to MockClusterer
    assert inspector.predict_clusters(X) == [1, 0]
    # Check if the df has not been modified by the prediction
    pd.testing.assert_frame_equal(X, X_copy)


@patch.object(MockClusterer, 'predict')
def test_predict_clusters__not_fitted(mock_clusterer):
    mock_clusterer.predict.return_value = [1, 0]

    # InspectorShap not fitted
    inspector = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=True)
    inspector.clusterer = mock_clusterer
    inspector.predicted_proba = True

    X = pd.DataFrame([1, 2, 3], [1, 2, 3])
    X_copy = pd.DataFrame([1, 2, 3], [1, 2, 3])

    # Check if not fitted exception is raised
    assert inspector.fitted is False
    with pytest.raises(NotFittedError):
        inspector.predict_clusters(X)
    # Check if X3 has not been modified
    pd.testing.assert_frame_equal(X, X_copy)


def test_assert_is_dataframe():
    X_list = [[1, 2, 3, 4], [1, 2, 3, 4]]
    X_df = pd.DataFrame(X_list)
    X_array = np.asarray(X_list)
    X_array_flat = np.asarray(X_list[0])


    assert X_df.equals(BaseInspector.assert_is_dataframe(X_df))
    assert X_df.equals(BaseInspector.assert_is_dataframe(X_array))
    with pytest.raises(NotImplementedError):
        BaseInspector.assert_is_dataframe(X_list)
    with pytest.raises(NotImplementedError):
        BaseInspector.assert_is_dataframe(X_array_flat)


def test_assert_is_series():
    X_list = [[1, 2, 3, 4], [1, 2, 3, 4]]
    X_list_flat = [1, 2, 3, 4]

    X_df = pd.DataFrame(X_list)
    X_df_flat = pd.DataFrame(X_list_flat)

    X_series = pd.Series(X_list_flat)
    X_array = np.asarray(X_list)
    X_array_flat = np.asarray(X_list_flat)
    index = [0, 1, 2, 3]

    assert X_series.equals(BaseInspector.assert_is_series(X_series))
    assert X_series.equals(BaseInspector.assert_is_series(X_df_flat))
    assert X_series.equals(BaseInspector.assert_is_series(X_array_flat, index=index))

    with pytest.raises(TypeError):
        BaseInspector.assert_is_series(X_list)
    with pytest.raises(TypeError):
        BaseInspector.assert_is_series(X_list_flat)
    with pytest.raises(TypeError):
        BaseInspector.assert_is_series(X_df)
    with pytest.raises(TypeError):
        BaseInspector.assert_is_series(X_array)
    with pytest.raises(TypeError):
        BaseInspector.assert_is_series(X_array, index=[0, 1])
    with pytest.raises(TypeError):
        BaseInspector.assert_is_series(X_array_flat)


def test_get_cluster_mask():
    df = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [1, 3, 4]], columns=['cluster_id', 'col_a', 'col_b'])
    cluster_id_1 = 1
    cluster_id_2 = [1, 4]

    expected_indexes_1 = [0, 4]
    expected_indexes_2 = [0, 3, 4]

    assert df.iloc[expected_indexes_1].equals(df[InspectorShap.get_cluster_mask(df, cluster_id_1)])
    assert df.iloc[expected_indexes_2].equals(df[InspectorShap.get_cluster_mask(df, cluster_id_2)])


@patch('probatus.interpret.inspector.return_confusion_metric')
def test_create_summary_df(mocked_method):
    cluster_series = pd.Series([1, 2, 3, 4], name='cluster_id')
    y_series = pd.Series([0, 1, 1, 0])
    probas = pd.Series([0.1, 0.2, 0.7, 0.1], name='probas')

    mocked_method.return_value = pd.Series([0.1, 0.8, 0.3, 0.1])

    expected_output = pd.DataFrame([[1, 0, 0.1, 0.1], [2, 1, 0.2, 0.8], [3, 1, 0.7, 0.3], [4, 0, 0.1, 0.1]],
                                   columns=['cluster_id', 'target', 'probas', 'confusion'])
    output = InspectorShap.create_summary_df(cluster_series, y_series, probas, normalize=False)

    mocked_method.assert_called_with(y_series, probas, normalize=False)
    assert output.equals(expected_output)


def test_aggregate_summary_df():
    df = pd.DataFrame([[1, 0, 0.1, 0.1], [2, 1, 0.2, 0.8], [3, 1, 0.7, 0.3], [4, 0, 0.1, 0.1],
                       [1, 0, 0.1, 0.1], [2, 0, 0.3, 0.3], [3, 1, 0.7, 0.3], [4, 0, 0.1, 0.1]],
                      columns=['cluster_id', 'target', 'pred_proba', 'confusion'])
    expected_output = pd.DataFrame(
        [[1, 0, 2, 0, 0.1, 0.1], [2, 1, 2, 0.5, 0.55, 0.25], [3, 2, 2, 1, 0.3, 0.7], [4, 0, 2, 0, 0.1, 0.1]],
        columns=['cluster_id', 'total_label_1', 'total_entries',
                 'label_1_rate', 'average_confusion', 'average_pred_proba'])
    pd.set_option('display.max_columns', None)

    assert InspectorShap.aggregate_summary_df(df).equals(expected_output)

def test_get_report__report_done():
    inspector = InspectorShap(model= MockModel(), algotype='kmeans', cluster_probability=False)
    report_value = pd.DataFrame([[1, 2], [2, 3]], columns=['cluster_id', 'column_a'])
    inspector.cluster_report = report_value
    assert inspector.get_report().equals(report_value)


def test_get_report__single_df():
    inspector = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=False)
    inspector.hasmultiple_dfs = False
    report_value = pd.DataFrame([[1, 2], [2, 3]], columns=['cluster_id', 'column_a'])

    assert inspector.cluster_report is None

    def mock_compute_report(self):
        self.agg_summary_df = report_value

    with patch.object(InspectorShap, '_compute_report', mock_compute_report):
        assert inspector.get_report().equals(report_value)
        assert inspector.agg_summary_df.equals(report_value)
        assert inspector.cluster_report.equals(report_value)


def test_get_report__multiple_df():
    inspector = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=False)
    inspector.hasmultiple_dfs = True

    report_value = pd.DataFrame([[1, 2], [2, 3]], columns=['cluster_id', 'column_a'])
    inspector.agg_summary_dfs = [pd.DataFrame([[1, 3], [2, 3]], columns=['cluster_id', 'column_a']),
                                 pd.DataFrame([[1, 2], [2, 3]], columns=['cluster_id', 'column_a'])]

    expected_result = pd.DataFrame([[1, 2, 3, 2], [2, 3, 3, 3]],
                                   columns=['cluster_id', 'column_a', 'column_a_sample_1', 'column_a_sample_2'])

    assert inspector.cluster_report is None
    assert inspector.set_names is None

    def mock_compute_report(self):
        self.agg_summary_df = report_value

    with patch.object(InspectorShap, '_compute_report', mock_compute_report):
        assert inspector.get_report().equals(expected_result)
        assert inspector.cluster_report.equals(expected_result)
        assert inspector.agg_summary_df.equals(report_value)


def test_get_report__multiple_df_set_names():
    inspector = InspectorShap(model=MockModel(), algotype='kmeans', cluster_probability=False)
    inspector.hasmultiple_dfs = True
    inspector.set_names = ['suf1', 'suf2']

    report_value = pd.DataFrame([[1, 2], [2, 3]], columns=['cluster_id', 'column_a'])
    inspector.agg_summary_dfs = [pd.DataFrame([[1, 3], [2, 3]], columns=['cluster_id', 'column_a']),
                                 pd.DataFrame([[1, 2], [2, 3]], columns=['cluster_id', 'column_a'])]

    expected_result = pd.DataFrame([[1, 2, 3, 2], [2, 3, 3, 3]],
                                   columns=['cluster_id', 'column_a', 'column_a_suf1', 'column_a_suf2'])

    assert inspector.cluster_report is None

    def mock_compute_report(self):
        self.agg_summary_df = report_value

    with patch.object(InspectorShap, '_compute_report', mock_compute_report):
        assert inspector.get_report().equals(expected_result)
        assert inspector.cluster_report.equals(expected_result)
        assert inspector.agg_summary_df.equals(report_value)


def test_slice_cluster_no_inputs_not_complementary():
    inspector = InspectorShap(model= MockModel(), algotype='kmeans')
    summary = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]], columns=['cluster_id', 'column_a', 'column_b'])
    inspector.summary_df = summary
    inspector.cluster_report = summary
    inspector.X_shap = X_shap = pd.DataFrame([[0, 1, 0], [1, 0, 0], [0, 0, 1]], columns=['shap_1', 'shap_2', 'shap_3'])
    inspector.y = y = pd.Series([1, 0, 0])
    inspector.predicted_proba = predicted_proba = pd.Series([0.1, 0.2, 0.3])

    target_cluster_id = 1
    correct_mask = returned_mask =[True, False, False]
    inspector.get_cluster_mask.return_value = correct_mask

    def mock_get_report(self):
        # Should not be called
        assert False

    def mock_get_cluster_mask(self, summary_df, cluster_id):
        assert summary_df is summary
        assert cluster_id is target_cluster_id
        return returned_mask

    with patch.object(InspectorShap, 'get_report', mock_get_report):
        with patch.object(InspectorShap, 'get_cluster_mask', mock_get_cluster_mask):
            shap_out, y_out, pred_out = inspector.slice_cluster(target_cluster_id, complementary=False)

            assert shap_out.equals(X_shap[correct_mask])
            assert y_out.equals(y[correct_mask])
            assert pred_out.equals(predicted_proba[correct_mask])


def test_slice_cluster_inputs_complementary():
    inspector = InspectorShap(model= MockModel(), algotype='kmeans')
    summary = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]], columns=['cluster_id', 'column_a', 'column_b'])
    X_shap = pd.DataFrame([[0, 1, 0], [1, 0, 0], [0, 0, 1]], columns=['shap_1', 'shap_2', 'shap_3'])
    y = pd.Series([1, 0, 0])
    predicted_proba = pd.Series([0.1, 0.2, 0.3])

    target_cluster_id = 1
    correct_mask = np.array([False, True, True])
    returned_mask = np.array([True, False, False])
    inspector.get_cluster_mask.return_value = correct_mask

    assert inspector.cluster_report is None

    def mock_get_report(self):
        self.cluster_report = summary

    def mock_get_cluster_mask(self, summary_df, cluster_id):
        assert summary_df is summary
        assert cluster_id is target_cluster_id
        return returned_mask

    with patch.object(InspectorShap, 'get_report', mock_get_report):
        with patch.object(InspectorShap, 'get_cluster_mask', mock_get_cluster_mask):
            shap_out, y_out, pred_out = inspector.slice_cluster(target_cluster_id, summary_df=summary,
                                                                X_shap=X_shap, y=y, predicted_proba=predicted_proba,
                                                                complementary=True)

            assert shap_out.equals(X_shap[correct_mask])
            assert y_out.equals(y[correct_mask])
            assert pred_out.equals(predicted_proba[correct_mask])
            assert summary.equals(inspector.cluster_report)


def test_init_inspector():
    mock_model = MockModel()
    inspector = InspectorShap(model=mock_model, algotype='kmeans', confusion_metric = 'proba',
                              normalize_probability=True, cluster_probability = True)
    assert inspector.model is mock_model
    assert inspector.isinspected is False
    assert inspector.hasmultiple_dfs is False
    assert inspector.normalize_proba is True
    assert inspector.cluster_probabilities is True
    assert inspector.agg_summary_df is None
    assert inspector.set_names is None
    assert inspector.confusion_metric is 'proba'
    assert inspector.cluster_report is None
    assert inspector.y is None
    assert inspector.predicted_proba is None
    assert inspector.X_shap is None
    assert inspector.clusters is None
    assert inspector.algotype is 'kmeans'
    assert inspector.fitted is False
    assert len(inspector.X_shaps) is 0
    assert len(inspector.clusters_list) is 0
    assert len(inspector.ys) is 0
    assert len(inspector.predicted_probas) is 0


def test_init_inspector_error():
    with pytest.raises(NotImplementedError):
        InspectorShap(model=MockModel(), algotype='kmeans', confusion_metric='error')


def test_init_inspector_error():
    with pytest.raises(UnsupportedModelError):
        InspectorShap(model=MockModel(), algotype='error', confusion_metric='proba')


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
