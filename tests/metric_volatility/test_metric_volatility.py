from probatus.metric_volatility import (
    BaseVolatilityEstimator,
    TrainTestVolatility,
    SplitSeedVolatility,
    BootstrappedVolatility,
    get_metric,
    sample_data,
    check_sampling_input,
)
from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from probatus.stat_tests.distribution_statistics import DistributionStatistics
from probatus.utils import Scorer, NotFittedError
import os
import matplotlib.pyplot as plt
import matplotlib

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")


@pytest.fixture(scope="function")
def X_array():
    """
    Fixture.
    """
    return np.array([[2, 1], [3, 2], [4, 3], [1, 2], [1, 1]])


@pytest.fixture(scope="function")
def y_list():
    """
    Fixture.
    """
    return [1, 0, 0, 1, 1]


@pytest.fixture(scope="function")
def y_array(y_list):
    """
    Fixture.
    """
    return np.array(y_list)


@pytest.fixture(scope="function")
def X_df(X_array):
    """
    Fixture.
    """
    return pd.DataFrame(X_array, columns=["c1", "c2"])


@pytest.fixture(scope="function")
def y_series(y_list):
    """
    Fixture.
    """
    return pd.Series(y_list)


@pytest.fixture(scope="function")
def iteration_results():
    """
    Fixture.
    """
    iterations_cols = ["metric_name", "train_score", "test_score", "delta_score"]
    return pd.DataFrame(
        [
            ["roc_auc", 0.8, 0.7, 0.1],
            ["roc_auc", 0.7, 0.6, 0.1],
            ["roc_auc", 0.9, 0.8, 0.1],
            ["accuracy", 1, 0.9, 0.1],
            ["accuracy", 0.8, 0.7, 0.1],
            ["accuracy", 0.9, 0.8, 0.1],
        ],
        columns=iterations_cols,
    )


@pytest.fixture(scope="function")
def report():
    """
    Fixture.
    """
    report_cols = ["train_mean", "train_std", "test_mean", "test_std", "delta_mean", "delta_std"]
    report_index = ["roc_auc", "accuracy"]
    return pd.DataFrame(
        [[0.8, 0.08164, 0.7, 0.08164, 0.1, 0], [0.9, 0.08164, 0.8, 0.08164, 0.1, 0]],
        columns=report_cols,
        index=report_index,
    ).astype(float)


@pytest.fixture(scope="function")
def iterations_train():
    """
    Fixture.
    """
    return pd.Series([0.8, 0.7, 0.9], name="train_score")


@pytest.fixture(scope="function")
def iterations_test():
    """
    Fixture.
    """
    return pd.Series([0.7, 0.6, 0.8], name="test_score")


@pytest.fixture(scope="function")
def iterations_delta():
    """
    Fixture.
    """
    return pd.Series([0.1, 0.1, 0.1], name="delta_score")


def test_inits(mock_model):
    """
    Test.
    """
    vol1 = SplitSeedVolatility(
        mock_model,
        scoring=["accuracy", "roc_auc"],
        test_prc=0.3,
        n_jobs=2,
        stats_tests_to_apply=["ES", "KS"],
        random_state=1,
        iterations=20,
    )

    assert id(vol1.clf) == id(mock_model)
    assert vol1.test_prc == 0.3
    assert vol1.n_jobs == 2
    assert vol1.stats_tests_to_apply == ["ES", "KS"]
    assert vol1.random_state == 1
    assert vol1.iterations == 20
    assert len(vol1.stats_tests_objects) == 2
    assert len(vol1.scorers) == 2
    assert vol1.sample_train_test_split_seed is True

    vol2 = BootstrappedVolatility(mock_model, scoring="roc_auc", stats_tests_to_apply="KS", test_sampling_fraction=0.8)

    assert id(vol2.clf) == id(mock_model)
    assert vol2.stats_tests_to_apply == ["KS"]
    assert len(vol2.stats_tests_objects) == 1
    assert len(vol2.scorers) == 1
    assert vol2.sample_train_test_split_seed is False
    assert vol2.test_sampling_fraction == 0.8
    assert vol2.fitted is False
    assert vol2.iterations_results is None
    assert vol2.report is None


def test_base_fit(mock_model, X_df, y_series):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model, random_state=1)

    with patch("numpy.random.seed") as mock_seed:
        vol.fit(X_df, y_series)
        mock_seed.assert_called_with(1)

    assert vol.iterations_results is None
    assert vol.report is None
    assert vol.fitted is True


def test_compute(report, mock_model):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model)

    with pytest.raises(NotFittedError):
        vol.compute()

    vol.fit()
    with pytest.raises(ValueError):
        vol.compute()

    vol.report = report

    pd.testing.assert_frame_equal(vol.compute(), report)
    pd.testing.assert_frame_equal(vol.compute(metrics=["roc_auc"]), report.loc[["roc_auc"]])
    pd.testing.assert_frame_equal(vol.compute(metrics="roc_auc"), report.loc[["roc_auc"]])


def test_plot(report, mock_model, iterations_train, iterations_test, iterations_delta):
    """
    Test.
    """
    with patch.object(BaseVolatilityEstimator, "compute", return_value=report.loc[["roc_auc"]]) as mock_compute:
        with patch.object(
            BaseVolatilityEstimator,
            "_get_samples_to_plot",
            return_value=(iterations_train, iterations_test, iterations_delta),
        ) as mock_get_samples:

            vol = BaseVolatilityEstimator(mock_model)
            vol.fitted = True

            vol.plot(metrics="roc_auc")
            mock_compute.assert_called_with(metrics="roc_auc")
            mock_get_samples.assert_called_with(metric_name="roc_auc")


def test_get_samples_to_plot(mock_model, iteration_results, iterations_train, iterations_test, iterations_delta):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model)
    vol.fitted = True
    vol.iterations_results = iteration_results

    train, test, delta = vol._get_samples_to_plot(metric_name="roc_auc")
    pd.testing.assert_series_equal(train, iterations_train)
    pd.testing.assert_series_equal(test, iterations_test)
    pd.testing.assert_series_equal(delta, iterations_delta)


def test_create_report(mock_model, iteration_results, report):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model)
    vol.fitted = True
    vol.iterations_results = iteration_results

    vol._create_report()
    pd.testing.assert_frame_equal(vol.report, report, check_less_precise=3)


def test_compute_mean_std_from_runs(mock_model, iteration_results):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model)
    results = vol._compute_mean_std_from_runs(iteration_results[iteration_results["metric_name"] == "roc_auc"])
    expected_results = [0.8, 0.08164, 0.7, 0.08164, 0.1, 0]
    for idx, item in enumerate(results):
        assert pytest.approx(item, 0.01) == expected_results[idx]


def test_compute_stats_tests_values(mock_model, iteration_results):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model, stats_tests_to_apply=["KS"])

    with patch.object(DistributionStatistics, "compute", return_value=(0.1, 0.05)):
        stats = vol._compute_stats_tests_values(iteration_results)

    assert stats[0] == 0.1
    assert stats[1] == 0.05


def test_fit_compute(mock_model, report, X_df, y_series):
    """
    Test.
    """
    vol = BaseVolatilityEstimator(mock_model)

    with patch.object(BaseVolatilityEstimator, "fit") as mock_fit:
        with patch.object(BaseVolatilityEstimator, "compute", return_value=report) as mock_compute:
            result = vol.fit_compute(X_df, y_series)

            mock_fit.assert_called_with(X_df, y_series)
            mock_compute.assert_called_with()

    pd.testing.assert_frame_equal(result, report)


def test_fit_train_test_sample_seed(mock_model, X_df, y_series, iteration_results):
    """
    Test.
    """
    vol = TrainTestVolatility(mock_model, scoring="roc_auc", iterations=3, sample_train_test_split_seed=True)

    with patch.object(BaseVolatilityEstimator, "fit") as mock_base_fit:
        with patch.object(TrainTestVolatility, "_create_report") as mock_create_report:
            with patch(
                "probatus.metric_volatility.volatility.get_metric",
                side_effect=[iteration_results.iloc[[0]], iteration_results.iloc[[1]], iteration_results.iloc[[2]]],
            ):

                vol.fit(X_df, y_series)

                mock_base_fit.assert_called_once()
                mock_create_report.assert_called_once()

    pd.testing.assert_frame_equal(vol.iterations_results, iteration_results.iloc[[0, 1, 2]])


def test_get_metric(mock_model, X_df, y_series):
    """
    Test.
    """
    split_seed = 1
    test_prc = 0.6
    with patch(
        "probatus.metric_volatility.metric.train_test_split",
        return_value=(X_df.iloc[[0, 1, 2]], X_df.iloc[[3, 4]], y_series.iloc[[0, 1, 2]], y_series.iloc[[3, 4]]),
    ) as mock_split:
        with patch(
            "probatus.metric_volatility.metric.sample_data",
            side_effect=[(X_df.iloc[[0, 1, 1]], y_series.iloc[[0, 1, 1]]), (X_df.iloc[[3, 3]], y_series.iloc[[3, 3]])],
        ) as mock_sample:
            with patch.object(Scorer, "score", side_effect=[0.8, 0.7]):
                output = get_metric(
                    X_df,
                    y_series,
                    mock_model,
                    test_size=test_prc,
                    split_seed=split_seed,
                    scorers=[Scorer("roc_auc")],
                    train_sampling_type="bootstrap",
                    test_sampling_type="bootstrap",
                    train_sampling_fraction=1,
                    test_sampling_fraction=1,
                )
                mock_split.assert_called_once()
                mock_sample.assert_called()
                mock_model.fit.assert_called()

    expected_output = pd.DataFrame(
        [["roc_auc", 0.8, 0.7, 0.1]], columns=["metric_name", "train_score", "test_score", "delta_score"]
    )
    pd.testing.assert_frame_equal(expected_output, output)


def test_sample_data_no_sampling(X_df, y_series):
    """
    Test.
    """
    with patch("probatus.metric_volatility.utils.check_sampling_input") as mock_sampling_input:
        X_out, y_out = sample_data(X_df, y_series, sampling_type=None, sampling_fraction=1)
        mock_sampling_input.assert_called_once()
    pd.testing.assert_frame_equal(X_out, X_df)
    pd.testing.assert_series_equal(y_out, y_series)


def test_sample_data_bootstrap(X_df, y_series):
    """
    Test.
    """
    with patch("probatus.metric_volatility.utils.check_sampling_input") as mock_sampling_input:
        X_out, y_out = sample_data(X_df, y_series, sampling_type="bootstrap", sampling_fraction=0.8)
        mock_sampling_input.assert_called_once()
    assert X_out.shape == (4, 2)
    assert y_out.shape == (4,)


def test_sample_data_sample(X_df, y_series):
    """
    Test.
    """
    with patch("probatus.metric_volatility.utils.check_sampling_input") as mock_sampling_input:
        X_out, y_out = sample_data(X_df, y_series, sampling_type="subsample", sampling_fraction=1)
        mock_sampling_input.assert_called_once()
    pd.testing.assert_frame_equal(X_out, X_df)
    pd.testing.assert_series_equal(y_out, y_series)


def test_check_sampling_input(X_array, y_array):
    """
    Test.
    """
    with pytest.raises(ValueError):
        check_sampling_input("bootstrap", 0, "dataset")
    with pytest.raises(ValueError):
        check_sampling_input("subsample", 0, "dataset")
    with pytest.raises(ValueError):
        check_sampling_input("subsample", 1, "dataset")
    with pytest.raises(ValueError):
        check_sampling_input("subsample", 10, "dataset")
    with pytest.raises(ValueError):
        check_sampling_input("wrong_name", 0.5, "dataset")


def test_fit_compute_full_process(X_df, y_series):
    """
    Test.
    """
    clf = DecisionTreeClassifier()
    vol = TrainTestVolatility(
        clf, scoring=["roc_auc", "recall"], iterations=3, sample_train_test_split_seed=False, random_state=42
    )

    report = vol.fit_compute(X_df, y_series)
    assert report.shape == (2, 6)

    # Check if plot runs
    vol.plot(show=False)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_fit_compute_complex(complex_data, complex_lightgbm):
    """
    Test.
    """
    X, y = complex_data
    vol = TrainTestVolatility(
        complex_lightgbm,
        scoring="roc_auc",
        iterations=3,
        sample_train_test_split_seed=True,
        verbose=150,
        random_state=42,
    )

    report = vol.fit_compute(X, y)
    assert report.shape == (1, 6)

    # Check if plot runs
    vol.plot(show=False)
