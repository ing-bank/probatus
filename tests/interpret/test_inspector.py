from probatus.interpret.inspector import return_confusion_metric, InspectorShap, BaseInspector
from tests.mocks import MockClusterer, MockModel
from unittest.mock import patch
from probatus.utils import NotFittedError, UnsupportedModelError

import numpy as np
import pandas as pd
import pytest

test_sensitivity = 0.0000000001


@pytest.mark.skip(reason="Not currently implemented")
def test_after_implementation_completed():
    """
    Test.
    """

    @pytest.fixture(scope="function")
    def global_clusters():
        return pd.Series([1, 2, 3, 4, 1, 2, 3, 4], name="cluster_id")

    @pytest.fixture(scope="function")
    def global_clusters_eval_set():
        return [pd.Series([1, 2, 3], name="cluster_id"), pd.Series([1, 2, 3], name="cluster_id")]

    @pytest.fixture(scope="function")
    def global_y():
        return pd.Series([0, 1, 1, 0, 0, 0, 1, 0])

    @pytest.fixture(scope="function")
    def global_X():
        return pd.DataFrame([[0], [1], [1], [0], [0], [0], [1], [0]])

    @pytest.fixture(scope="function")
    def global_confusion_metric():
        return pd.Series([0.1, 0.8, 0.3, 0.1, 0.1, 0.3, 0.3, 0.1])

    @pytest.fixture(scope="function")
    def global_summary_df(columns_summary_df):
        return pd.DataFrame(
            [
                [1, 0, 0.1, 0.1],
                [2, 1, 0.2, 0.8],
                [3, 1, 0.7, 0.3],
                [4, 0, 0.1, 0.1],
                [1, 0, 0.1, 0.1],
                [2, 0, 0.3, 0.3],
                [3, 1, 0.7, 0.3],
                [4, 0, 0.1, 0.1],
            ],
            columns=columns_summary_df,
        )

    @pytest.fixture(scope="function")
    def global_X_shap():
        return pd.DataFrame(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 2],
            ],
            columns=["shap1", "shap2", "shap3", "shap4"],
        )

    @pytest.fixture(scope="function")
    def columns_aggregate_summary_df():
        return [
            "cluster_id",
            "total_label_1",
            "total_entries",
            "label_1_rate",
            "average_confusion",
            "average_pred_proba",
        ]

    @pytest.fixture(scope="function")
    def columns_summary_df():
        return ["cluster_id", "target", "pred_proba", "confusion"]

    @pytest.fixture(scope="function")
    def global_aggregate_summary_df(columns_aggregate_summary_df):
        return pd.DataFrame(
            [[1, 0, 2, 0, 0.1, 0.1], [2, 1, 2, 0.5, 0.55, 0.25], [3, 2, 2, 1, 0.3, 0.7], [4, 0, 2, 0, 0.1, 0.1]],
            columns=columns_aggregate_summary_df,
        )

    @pytest.fixture(scope="function")
    def global_aggregate_summary_dfs_eval_set(columns_aggregate_summary_df):
        return [
            pd.DataFrame(
                [[1, 0, 1, 0, 0.1, 0.1], [2, 0, 1, 0, 0.2, 0.2], [3, 0, 1, 0, 0.3, 0.3]],
                columns=columns_aggregate_summary_df,
            ),
            pd.DataFrame(
                [[1, 1, 1, 1, 0.4, 0.6], [2, 1, 1, 1, 0.5, 0.5], [3, 1, 1, 1, 0.6, 0.4]],
                columns=columns_aggregate_summary_df,
            ),
        ]

    @pytest.fixture(scope="function")
    def global_summary_dfs():
        return [
            pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]], columns=["cluster_id", "column_a", "column_b"]),
            pd.DataFrame([[1, 2, 1], [2, 3, 2], [3, 4, 3]], columns=["cluster_id", "column_a", "column_b"]),
        ]

    @pytest.fixture(scope="function")
    def global_X_shaps():
        return [
            pd.DataFrame([[0, 3, 0], [3, 0, 0], [0, 0, 3]], columns=["shap_1", "shap_2", "shap_3"]),
            pd.DataFrame([[0, 2, 0], [2, 0, 0], [0, 0, 2]], columns=["shap_1", "shap_2", "shap_3"]),
        ]

    @pytest.fixture(scope="function")
    def global_ys():
        return [pd.Series([0, 0, 0]), pd.Series([1, 1, 1])]

    @pytest.fixture(scope="function")
    def global_Xs():
        return [pd.DataFrame([[0], [1], [1]]), pd.DataFrame([[0], [1], [1]])]

    @pytest.fixture(scope="function")
    def global_predicted_probas():
        return [pd.Series([0.1, 0.2, 0.3]), pd.Series([0.4, 0.5, 0.6])]

    @pytest.fixture(scope="function")
    def global_predicted_proba():
        return pd.Series([0.1, 0.2, 0.7, 0.1, 0.1, 0.3, 0.7, 0.1], name="pred_proba")

    @pytest.fixture(scope="function")
    def global_small_df():
        return pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]])

    @pytest.fixture(scope="function")
    def global_small_df_flat():
        return pd.Series([1, 2, 3, 4])

    @pytest.fixture(scope="function")
    def global_mock_aggregate_summary_dfs():
        return [
            pd.DataFrame([[1, 3], [2, 3]], columns=["cluster_id", "column_a"]),
            pd.DataFrame([[1, 2], [2, 3]], columns=["cluster_id", "column_a"]),
        ]

    @pytest.fixture(scope="function")
    def global_mock_summary_df():
        return pd.DataFrame([[1, 2], [2, 3]], columns=["cluster_id", "column_a"])

    def test_return_confusion_metric__array():
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)

        expected_output_not_normalized = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], dtype=float)
        expected_output_normalized = np.array(
            [0.11111111, 0.22222222, 0.33333333, 0.22222222, 0.11111111, 0.0], dtype=float
        )
        assert (
            expected_output_normalized - return_confusion_metric(y_true, y_score, normalize=True < test_sensitivity)
        ).all()
        assert (
            expected_output_not_normalized - return_confusion_metric(y_true, y_score, normalize=False)
            < test_sensitivity
        ).all()

    def test_return_confusion_metric__series():
        # The method also needs to work with series, since it is called with series by create summary df
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        y_score = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        expected_output_not_normalized = pd.Series([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], dtype=float)
        expected_output_normalized = pd.Series(
            [0.11111111, 0.22222222, 0.33333333, 0.22222222, 0.11111111, 0.0], dtype=float
        )
        assert (
            expected_output_normalized - return_confusion_metric(y_true, y_score, normalize=True < test_sensitivity)
        ).all()
        assert (
            expected_output_not_normalized - return_confusion_metric(y_true, y_score, normalize=False)
            < test_sensitivity
        ).all()

    @patch.object(MockClusterer, "fit")
    def test_fit_clusters__base_inspector(mock_clusterer, global_small_df):
        # Base Inspector case algotype is kmeans
        inspector = BaseInspector(algotype="kmeans")
        inspector.clusterer = mock_clusterer

        X = global_small_df

        inspector.fit_clusters(X)

        # Check if has been called with correct argument
        mock_clusterer.fit.assert_called_once()
        pd.testing.assert_frame_equal(mock_clusterer.fit.call_args[0][0], X)
        # Check if it has not been modified
        pd.testing.assert_frame_equal(X, global_small_df)
        # Check if fitted flag has been changed correctly
        assert inspector.fitted is True

    @patch.object(MockClusterer, "fit")
    def test_fit_clusters__inspector_shap(mock_clusterer, global_small_df):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=False)
        inspector.clusterer = mock_clusterer

        X = global_small_df

        inspector.fit_clusters(X)

        # Check if has been called with correct argument
        mock_clusterer.fit.assert_called_once()
        pd.testing.assert_frame_equal(mock_clusterer.fit.call_args[0][0], X)
        # Check if it has not been modified
        pd.testing.assert_frame_equal(X, global_small_df)
        # Check if fitted flag has been changed correctly
        assert inspector.fitted is True

    @patch.object(MockClusterer, "fit")
    def test_fit_clusters__inspector_shap_proba(mock_clusterer, global_small_df):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=True)
        inspector.clusterer = mock_clusterer
        inspector.predicted_proba = True

        X = global_small_df

        # Check if not fitted exception is raised
        inspector.fit_clusters(X)

        # Check if column with probabilities has been added to the fitted X
        assert "probs" in mock_clusterer.fit.call_args[0][0].columns

        # Check if has been called
        mock_clusterer.fit.assert_called_once()

        # Check if X has not been modified
        pd.testing.assert_frame_equal(X, global_small_df)
        assert inspector.fitted is True

    @patch.object(MockClusterer, "predict")
    def test_predict_clusters__base_inspector(mock_clusterer, global_small_df):
        mock_clusterer.predict.return_value = [1, 0]

        inspector = BaseInspector(algotype="kmeans")
        inspector.clusterer = mock_clusterer
        inspector.fitted = True

        X = global_small_df

        # Check if the prediction is correct according to the Mock clusterer
        assert inspector.predict_clusters(X) == [1, 0]

        # Check if the clusterer was called with correct input
        mock_clusterer.predict.assert_called_once()
        pd.testing.assert_frame_equal(mock_clusterer.predict.call_args[0][0], X)

        # Check if the X has not been modified
        pd.testing.assert_frame_equal(X, global_small_df)

    @patch.object(MockClusterer, "predict")
    def test_predict_clusters__inspector_shap(mock_clusterer, global_small_df):
        mock_clusterer.predict.return_value = [1, 0]

        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=False)
        inspector.clusterer = mock_clusterer
        inspector.fitted = True

        X = global_small_df

        # Check if the output is correct, as should be according to MockClusterer
        assert inspector.predict_clusters(X) == [1, 0]
        # Check if the df has not been modified by the prediction
        pd.testing.assert_frame_equal(X, global_small_df)

    @patch.object(MockClusterer, "predict")
    def test_predict_clusters__not_fitted(mock_clusterer, global_small_df):
        mock_clusterer.predict.return_value = [1, 0]

        # InspectorShap not fitted
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=True)
        inspector.clusterer = mock_clusterer
        inspector.predicted_proba = True

        X = global_small_df

        # Check if not fitted exception is raised
        with pytest.raises(NotFittedError):
            inspector.predict_clusters(X)
        # Check if X3 has not been modified
        pd.testing.assert_frame_equal(X, global_small_df)

    def test_assert_is_dataframe(global_small_df):
        X_df = global_small_df
        X_list = X_df.values.tolist()
        X_array = np.asarray(X_list)
        X_array_flat = np.asarray(X_list[0])

        pd.testing.assert_frame_equal(X_df, BaseInspector.assert_is_dataframe(X_df))
        pd.testing.assert_frame_equal(X_df, BaseInspector.assert_is_dataframe(X_array))
        with pytest.raises(NotImplementedError):
            BaseInspector.assert_is_dataframe(X_list)
        with pytest.raises(NotImplementedError):
            BaseInspector.assert_is_dataframe(X_array_flat)

    def test_assert_is_series(global_small_df, global_small_df_flat):
        X_df = global_small_df
        X_df_flat = global_small_df_flat
        X_list = X_df.values.tolist()
        X_list_flat = X_df_flat.values.tolist()

        X_series = pd.Series(X_list_flat)
        X_array = np.asarray(X_list)
        X_array_flat = np.asarray(X_list_flat)
        index = [0, 1, 2, 3]

        pd.testing.assert_series_equal(X_series, BaseInspector.assert_is_series(X_series))
        pd.testing.assert_series_equal(X_series, BaseInspector.assert_is_series(X_df_flat))
        pd.testing.assert_series_equal(X_series, BaseInspector.assert_is_series(X_array_flat, index=index))

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

    def test_get_cluster_mask(global_summary_df):
        df = global_summary_df
        cluster_id_1 = 1
        cluster_id_2 = [1, 4]

        expected_indexes_1 = [0, 4]
        expected_indexes_2 = [0, 3, 4, 7]

        pd.testing.assert_frame_equal(df.iloc[expected_indexes_1], df[InspectorShap.get_cluster_mask(df, cluster_id_1)])
        pd.testing.assert_frame_equal(df.iloc[expected_indexes_2], df[InspectorShap.get_cluster_mask(df, cluster_id_2)])

    @patch("probatus.interpret.inspector.return_confusion_metric")
    def test_create_summary_df(
        mocked_method, global_clusters, global_y, global_predicted_proba, global_confusion_metric, global_summary_df
    ):
        cluster_series = global_clusters
        y_series = global_y
        probas = global_predicted_proba

        mocked_method.return_value = global_confusion_metric
        expected_output = global_summary_df

        output = InspectorShap.create_summary_df(cluster_series, y_series, probas, normalize=False)

        # Check if method is called with correct input
        mocked_method.assert_called_once()
        pd.testing.assert_series_equal(mocked_method.call_args[0][0], y_series)
        pd.testing.assert_series_equal(mocked_method.call_args[0][1], probas)
        assert mocked_method.call_args_list[0][1]["normalize"] is False

        # Check if the output is correct
        pd.testing.assert_frame_equal(output, expected_output)

    def test_aggregate_summary_df(global_summary_df, global_aggregate_summary_df):
        df = global_summary_df
        expected_output = global_aggregate_summary_df
        pd.set_option("display.max_columns", None)

        pd.testing.assert_frame_equal(InspectorShap.aggregate_summary_df(df), expected_output)

    def test_compute__report_done():
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=False)
        report_value = pd.DataFrame([[1, 2], [2, 3]], columns=["cluster_id", "column_a"])
        inspector.cluster_report = report_value

        pd.testing.assert_frame_equal(inspector.compute(), report_value)

    def test_compute__single_df(global_mock_summary_df):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=False)
        inspector.hasmultiple_dfs = False

        report_value = global_mock_summary_df

        def mock_compute_report(self):
            self.agg_summary_df = report_value

        with patch.object(InspectorShap, "_compute_report", mock_compute_report):
            output = inspector.compute()

        # Check output and side effects
        pd.testing.assert_frame_equal(output, report_value)
        pd.testing.assert_frame_equal(inspector.cluster_report, report_value)
        pd.testing.assert_frame_equal(inspector.agg_summary_df, report_value)

    def test_compute__multiple_df(global_mock_summary_df, global_mock_aggregate_summary_dfs):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=False)
        inspector.hasmultiple_dfs = True

        report_value = global_mock_summary_df
        inspector.agg_summary_dfs = global_mock_aggregate_summary_dfs

        expected_result = pd.DataFrame(
            [[1, 2, 3, 2], [2, 3, 3, 3]], columns=["cluster_id", "column_a", "column_a_sample_1", "column_a_sample_2"]
        )

        def mock_compute_report(self):
            self.agg_summary_df = report_value

        with patch.object(InspectorShap, "_compute_report", mock_compute_report):
            output = inspector.compute()

        # Check output and side effects
        pd.testing.assert_frame_equal(output, expected_result)
        pd.testing.assert_frame_equal(inspector.cluster_report, expected_result)
        pd.testing.assert_frame_equal(inspector.agg_summary_df, report_value)

    def test_compute__multiple_df_set_names(global_mock_summary_df, global_mock_aggregate_summary_dfs):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans", cluster_probability=False)
        inspector.hasmultiple_dfs = True
        inspector.set_names = ["suf1", "suf2"]

        report_value = global_mock_summary_df
        inspector.agg_summary_dfs = global_mock_aggregate_summary_dfs

        expected_result = pd.DataFrame(
            [[1, 2, 3, 2], [2, 3, 3, 3]], columns=["cluster_id", "column_a", "column_a_suf1", "column_a_suf2"]
        )

        def mock_compute_report(self):
            self.agg_summary_df = report_value

        with patch.object(InspectorShap, "_compute_report", mock_compute_report):
            output = inspector.compute()

        # Check output and side effects
        pd.testing.assert_frame_equal(output, expected_result)
        pd.testing.assert_frame_equal(inspector.cluster_report, expected_result)
        pd.testing.assert_frame_equal(inspector.agg_summary_df, report_value)

    def test_slice_cluster_no_inputs_not_complementary(
        global_summary_df, global_X_shap, global_y, global_predicted_proba
    ):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        summary = global_summary_df
        inspector.summary_df = summary
        inspector.cluster_report = summary
        inspector.X_shap = X_shap = global_X_shap
        inspector.y = y = global_y
        inspector.predicted_proba = predicted_proba = global_predicted_proba

        target_cluster_id = 1
        correct_mask = returned_mask = [True, False, False, False, True, False, False, False]
        inspector.get_cluster_mask.return_value = correct_mask

        with patch.object(InspectorShap, "compute") as mocked_compute:
            with patch.object(InspectorShap, "get_cluster_mask") as mock_get_cluster_mask:
                mock_get_cluster_mask.return_value = returned_mask
                shap_out, y_out, pred_out = inspector.slice_cluster(target_cluster_id, complementary=False)

                # Ensure mocked_compute not called
                mocked_compute.accert_not_called()
                # Ensure mock_get_cluster_mask called with correct arguments
                mock_get_cluster_mask.assert_called_once()
                pd.testing.assert_frame_equal(mock_get_cluster_mask.call_args[0][0], summary)
                assert mock_get_cluster_mask.call_args[0][1] == target_cluster_id

        # Check outputs
        pd.testing.assert_frame_equal(shap_out, X_shap[correct_mask])
        pd.testing.assert_series_equal(y_out, y[correct_mask])
        pd.testing.assert_series_equal(pred_out, predicted_proba[correct_mask])

    def test_slice_cluster_inputs_complementary(global_summary_df, global_X_shap, global_y, global_predicted_proba):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        summary = global_summary_df
        X_shap = global_X_shap
        y = global_y
        predicted_proba = global_predicted_proba

        target_cluster_id = 1
        correct_mask = np.array([False, True, True, True, False, True, True, True])
        returned_mask = np.logical_not(correct_mask)
        inspector.get_cluster_mask.return_value = correct_mask

        assert inspector.cluster_report is None

        def mock_compute(self):
            self.cluster_report = summary

        with patch.object(InspectorShap, "compute", mock_compute):
            with patch.object(InspectorShap, "get_cluster_mask") as mock_get_cluster_mask:
                mock_get_cluster_mask.return_value = returned_mask
                shap_out, y_out, pred_out = inspector.slice_cluster(
                    target_cluster_id,
                    summary_df=summary,
                    X_shap=X_shap,
                    y=y,
                    predicted_proba=predicted_proba,
                    complementary=True,
                )
                # Ensure mocked_get_cluster_mask called with correct arguments
                mock_get_cluster_mask.assert_called_once()
                pd.testing.assert_frame_equal(mock_get_cluster_mask.call_args[0][0], summary)
                assert mock_get_cluster_mask.call_args[0][1] == target_cluster_id

        # Check outputs and side effects
        pd.testing.assert_frame_equal(shap_out, X_shap[correct_mask])
        pd.testing.assert_series_equal(y_out, y[correct_mask])
        pd.testing.assert_series_equal(pred_out, predicted_proba[correct_mask])
        pd.testing.assert_frame_equal(inspector.cluster_report, summary)

    def test_init_inspector():
        mock_model = MockModel()
        inspector = InspectorShap(
            model=mock_model,
            algotype="kmeans",
            confusion_metric="proba",
            normalize_probability=True,
            cluster_probability=True,
        )
        assert inspector.model is mock_model
        assert inspector.isinspected is False
        assert inspector.hasmultiple_dfs is False
        assert inspector.normalize_proba is True
        assert inspector.cluster_probabilities is True
        assert inspector.agg_summary_df is None
        assert inspector.set_names is None
        assert inspector.confusion_metric == "proba"
        assert inspector.cluster_report is None
        assert inspector.y is None
        assert inspector.predicted_proba is None
        assert inspector.X_shap is None
        assert inspector.clusters is None
        assert inspector.algotype == "kmeans"
        assert inspector.fitted is False
        assert inspector.X_shaps == list()
        assert inspector.clusters_list == list()
        assert inspector.ys == list()
        assert inspector.predicted_probas == list()

    def test_init_inspector_error():
        with pytest.raises(NotImplementedError):
            InspectorShap(model=MockModel(), algotype="kmeans", confusion_metric="error")

    def test_init_inspector_error2():
        with pytest.raises(UnsupportedModelError):
            InspectorShap(model=MockModel(), algotype="error", confusion_metric="proba")

    def test_slice_cluster_eval_sets__single_df():
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        inspector.hasmultiple_dfs = False
        cluster_id = 1
        with pytest.raises(NotFittedError):
            inspector.slice_cluster_eval_set(cluster_id)

    def test_slice_cluster_eval_sets__multiple_df(
        global_X_shaps, global_ys, global_predicted_probas, global_summary_dfs
    ):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        inspector.hasmultiple_dfs = True

        inspector.X_shaps = X_shaps = global_X_shaps
        inspector.ys = ys = global_ys
        inspector.predicted_probas = predicted_probas = global_predicted_probas
        inspector.summary_dfs = summary_dfs = global_summary_dfs

        target_row = [0]
        target_cluster_id = 1
        target_complementary = False

        target_output = [
            [pd.DataFrame([[0, 3, 0]], columns=["shap_1", "shap_2", "shap_3"]), pd.Series([0]), pd.Series([0.1])],
            [pd.DataFrame([[0, 2, 0]], columns=["shap_1", "shap_2", "shap_3"]), pd.Series([1]), pd.Series([0.4])],
        ]

        with patch.object(InspectorShap, "slice_cluster") as mock_slice_cluster:
            # Setting multiple outputs
            mock_slice_cluster.side_effect = [
                (X_shaps[0].iloc[target_row], ys[0].iloc[target_row], predicted_probas[0].iloc[target_row]),
                (X_shaps[1].iloc[target_row], ys[1].iloc[target_row], predicted_probas[1].iloc[target_row]),
            ]

            output = inspector.slice_cluster_eval_set(target_cluster_id, complementary=target_complementary)

            # Check if inputs are correct at each call
            for call_index, call in enumerate(mock_slice_cluster.call_args_list):
                # On the position 1 of call there are kwargs
                assert call[1]["cluster_id"] == target_cluster_id
                assert call[1]["complementary"] == target_complementary
                pd.testing.assert_frame_equal(call[1]["summary_df"], summary_dfs[call_index])
                pd.testing.assert_frame_equal(call[1]["X_shap"], X_shaps[call_index])
                pd.testing.assert_series_equal(call[1]["predicted_proba"], predicted_probas[call_index])
                pd.testing.assert_series_equal(call[1]["y"], ys[call_index])

        # Check lengths of lists
        assert len(output) is len(target_output)

        # Go over the output and check each element
        for index, current_output in enumerate(output):
            pd.testing.assert_frame_equal(target_output[index][0], current_output[0])
            pd.testing.assert_series_equal(target_output[index][1], current_output[1])
            pd.testing.assert_series_equal(target_output[index][2], current_output[2])

    def test_compute_report_single_df(
        global_clusters, global_y, global_predicted_proba, global_summary_df, global_aggregate_summary_df
    ):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        inspector.hasmultiple_dfs = False
        inspector.normalize_proba = target_normalize = False

        inspector.clusters = input_clust = global_clusters
        inspector.y = input_y = global_y
        inspector.predicted_proba = input_predicted_proba = global_predicted_proba
        target_summary_df = global_summary_df
        aggregated_summary = global_aggregate_summary_df

        with patch.object(InspectorShap, "create_summary_df") as mock_create_summary_df:
            with patch.object(InspectorShap, "aggregate_summary_df") as mock_aggregate_summary_df:
                mock_create_summary_df.return_value = target_summary_df
                mock_aggregate_summary_df.return_value = aggregated_summary

                inspector._compute_report()

                # check if the methods were called with correct arguments
                pd.testing.assert_frame_equal(mock_aggregate_summary_df.call_args[0][0], target_summary_df)
                pd.testing.assert_series_equal(mock_create_summary_df.call_args[0][0], input_clust)
                pd.testing.assert_series_equal(mock_create_summary_df.call_args[0][1], input_y)
                pd.testing.assert_series_equal(mock_create_summary_df.call_args[0][2], input_predicted_proba)
                assert mock_create_summary_df.call_args[1]["normalize"] == target_normalize

        # Check if the function correctly stored variables
        pd.testing.assert_frame_equal(inspector.agg_summary_df, aggregated_summary)
        pd.testing.assert_frame_equal(inspector.summary_df, target_summary_df)

    def test_compute_report_multiple_df(
        global_clusters,
        global_y,
        global_predicted_proba,
        global_summary_df,
        global_aggregate_summary_df,
        global_summary_dfs,
        global_ys,
        global_predicted_probas,
        global_aggregate_summary_dfs_eval_set,
    ):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        inspector.hasmultiple_dfs = True
        inspector.normalize_proba = False

        inspector.clusters = global_clusters
        inspector.y = global_y
        inspector.predicted_proba = global_predicted_proba
        inspector.ys = global_ys
        inspector.predicted_probas = global_predicted_probas
        target_summary_df = global_summary_df
        target_summary_dfs = global_summary_dfs
        aggregated_summary_df = global_aggregate_summary_df
        aggregated_summary_dfs = global_aggregate_summary_dfs_eval_set

        with patch.object(InspectorShap, "create_summary_df") as mock_create_summary_df:
            with patch.object(InspectorShap, "aggregate_summary_df") as mock_aggregate_summary_df:
                # Set returns for each call of methods
                mock_create_summary_df.side_effect = [target_summary_df, target_summary_dfs[0], target_summary_dfs[1]]
                mock_aggregate_summary_df.side_effect = [
                    aggregated_summary_df,
                    aggregated_summary_dfs[0],
                    aggregated_summary_dfs[1],
                ]
                inspector._compute_report()

        assert inspector.agg_summary_df.equals(aggregated_summary_df)
        assert inspector.summary_df.equals(target_summary_df)
        for index, item in inspector.agg_summary_dfs:
            assert item.equals(aggregated_summary_dfs[index])
        for index, item in inspector.summary_dfs:
            assert item.equals(target_summary_dfs[index])

    def test_perform_fit_calc(global_X, global_y, global_predicted_proba, global_X_shap, global_clusters):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        inspector.model = MockModel()
        input_X = global_X
        input_y = global_y
        input_predicted_proba = global_predicted_proba
        values_probabilities = input_predicted_proba.tolist()

        def mock_fit_clusters(self, X_shap):
            inspector.fitted = True

        with patch.object(InspectorShap, "assert_is_dataframe") as mock_assert_is_dataframe:
            with patch.object(InspectorShap, "assert_is_series") as mock_assert_is_series:
                with patch.object(InspectorShap, "compute_probabilities") as mock_compute_probabilities:
                    with patch("probatus.interpret._shap_helpers.shap_to_df") as mock_shap_to_df:
                        with patch.object(InspectorShap, "fit_clusters", mock_fit_clusters):
                            with patch.object(InspectorShap, "predict_clusters") as mock_predict_clusters:
                                mock_assert_is_dataframe.return_value = input_X
                                mock_assert_is_series.return_value = input_y
                                mock_compute_probabilities.return_value = values_probabilities
                                mock_shap_to_df.return_value = global_X_shap
                                mock_predict_clusters.return_value = global_clusters.tolist()

                                out_y, out_predicted_proba, out_X_shap, out_clusters = inspector.perform_fit_calc(
                                    input_X, input_y, fit_clusters=True
                                )

        pd.testing.assert_series_equal(out_y, input_y)
        pd.testing.assert_series_equal(out_predicted_proba, input_predicted_proba)
        pd.testing.assert_frame_equal(out_X_shap, global_X_shap)
        pd.testing.assert_series_equal(out_clusters, global_clusters)
        assert inspector.fitted is True

    def test_fit__multiple_df(
        global_X,
        global_y,
        global_predicted_proba,
        global_X_shap,
        global_clusters,
        global_Xs,
        global_ys,
        global_predicted_probas,
        global_clusters_eval_set,
        global_X_shaps,
    ):

        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        input_eval_set = [(global_Xs[0], global_ys[0]), (global_Xs[1], global_ys[1])]
        input_sample_names = ["set1", "set2"]
        input_X = global_X
        input_y = global_y

        with patch.object(InspectorShap, "perform_fit_calc") as mock_perform_fit_calc:
            with patch.object(InspectorShap, "init_eval_set_report_variables") as mock_init_variables:
                mock_perform_fit_calc.side_effect = [
                    (global_y, global_predicted_proba, global_X_shap, global_clusters),
                    (global_ys[0], global_predicted_probas[0], global_X_shaps[0], global_clusters_eval_set[0]),
                    (global_ys[1], global_predicted_probas[1], global_X_shaps[1], global_clusters_eval_set[1]),
                ]

                inspector.fit(X=input_X, y=input_y, eval_set=input_eval_set, sample_names=input_sample_names)
                mock_init_variables.assert_called_once()

        assert inspector.hasmultiple_dfs is True
        assert inspector.set_names is input_sample_names
        assert inspector.y.equals(global_y)
        assert inspector.predicted_proba.equals(global_predicted_proba)
        assert inspector.X_shap.equals(global_X_shap)
        assert inspector.clusters.equals(global_clusters)
        assert all([a.equals(b) for a, b in zip(inspector.clusters_list, global_clusters_eval_set)])
        assert all([a.equals(b) for a, b in zip(inspector.X_shaps, global_X_shaps)])
        assert all([a.equals(b) for a, b in zip(inspector.predicted_probas, global_predicted_probas)])
        assert all([a.equals(b) for a, b in zip(inspector.ys, global_ys)])
        assert input_sample_names is inspector.set_names

    def test_compute_probabilities(global_X):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        input_X = global_X
        model_probas = np.array(
            [[0.2, 0.8], [0.7, 0.3], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3], [0.5, 0.5]]
        )
        expected_output = np.array([0.8, 0.3, 0.3, 0.7, 0.8, 0.7, 0.3, 0.5])

        with patch.object(MockModel, "predict_proba") as mock_predict_proba:
            mock_predict_proba.return_value = model_probas
            np.testing.assert_array_equal(expected_output, inspector.compute_probabilities(input_X))

    def test_fit_compute(global_X, global_aggregate_summary_df):
        inspector = InspectorShap(model=MockModel(), algotype="kmeans")
        input_X = global_X
        expected_output = global_aggregate_summary_df

        with patch.object(InspectorShap, "fit") as mock_fit:
            with patch.object(InspectorShap, "compute") as mock_compute:
                mock_compute.return_value = global_aggregate_summary_df

                output = inspector.fit_compute(input_X)

                # Check if fit called with input X
                pd.testing.assert_frame_equal(mock_fit.call_args[0][0], input_X)
        # Check if the returned value correct
        pd.testing.assert_frame_equal(expected_output, output)
