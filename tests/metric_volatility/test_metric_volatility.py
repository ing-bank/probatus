from probatus.metric_volatility import BaseVolatilityEstimator, TrainTestVolatility, SplitSeedVolatility,\
    BootstrappedVolatility
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from probatus.utils import NotFittedError
import matplotlib.pyplot as plt
from probatus.stat_tests import DistributionStatistics

@pytest.fixture(scope='function')
def X_array():
    return np.array([[2, 1], [3, 2], [4, 3], [1, 2], [1, 1]])


@pytest.fixture(scope='function')
def y_list():
    return [1, 0, 0, 1, 1]


@pytest.fixture(scope='function')
def y_array(y_list):
    return np.array(y_list)


@pytest.fixture(scope='function')
def X_df(X_array):
    return pd.DataFrame(X_array, columns=['c1', 'c2'])


@pytest.fixture(scope='function')
def y_series(y_list):
    return pd.DataFrame(y_list)

@pytest.fixture(scope='function')
def iteration_results():
    iterations_cols = ['metric_name', 'train_score', 'test_score', 'delta_score']
    return pd.DataFrame([['roc_auc', 0.8, 0.7, 0.1], ['roc_auc', 0.7, 0.6, 0.1], ['roc_auc', 0.9, 0.8, 0.1],
                         ['accuracy', 1, 0.9, 0.1], ['accuracy', 0.8, 0.7, 0.1], ['accuracy', 0.9, 0.8, 0.1]],
                        columns=iterations_cols)

@pytest.fixture(scope='function')
def report():
    report_cols = ['train_mean', 'train_std', 'test_mean', 'test_std', 'delta_mean', 'delta_std']
    report_index = ['roc_auc', 'accuracy']
    return pd.DataFrame([[0.8, 0.08164, 0.7, 0.08164, 0.1, 0],
                         [0.9, 0.08164, 0.8, 0.08164, 0.1, 0]], columns=report_cols, index=report_index).astype(float)

@pytest.fixture(scope='function')
def iterations_train():
    return pd.Series([0.8, 0.7, 0.9], name='train_score')

@pytest.fixture(scope='function')
def iterations_test():
    return pd.Series([0.7, 0.6, 0.8], name='test_score')

@pytest.fixture(scope='function')
def iterations_delta():
    return pd.Series([0.1, 0.1, 0.1], name='delta_score')

def test_inits(mock_model):
    vol1 = SplitSeedVolatility(mock_model, metrics=['accuracy', 'roc_auc'], test_prc=0.3, n_jobs=2,
                              stats_tests_to_apply=['ES', 'KS'], random_state=1, iterations=20)

    assert id(vol1.model) == id(mock_model)
    assert vol1.test_prc == 0.3
    assert vol1.n_jobs == 2
    assert vol1.stats_tests_to_apply == ['ES', 'KS']
    assert vol1.random_state == 1
    assert vol1.iterations == 20
    assert len(vol1.stats_tests_objects) == 2
    assert len(vol1.scorers) == 2
    assert vol1.sample_train_test_split_seed is True

    vol2 = BootstrappedVolatility(mock_model, metrics='roc_auc', stats_tests_to_apply='KS', test_sampling_fraction=0.8)

    assert id(vol2.model) == id(mock_model)
    assert vol2.stats_tests_to_apply == ['KS']
    assert len(vol2.stats_tests_objects) == 1
    assert len(vol2.scorers) == 1
    assert vol2.sample_train_test_split_seed is False
    assert vol2.test_sampling_fraction == 0.8
    assert vol2.fitted is False
    assert vol2.iterations_results is None
    assert vol2.report is None

def test_base_fit(mock_model, X_df, y_series):
    vol = BaseVolatilityEstimator(mock_model, random_state=1)

    with patch('numpy.random.seed') as mock_seed:
        vol.fit(X_df, y_series)
        mock_seed.assert_called_with(1)

    assert vol.iterations_results is None
    assert vol.report is None
    assert vol.fitted is True


def test_compute(report, mock_model, iterations_train, iterations_test, iterations_delta):
    vol = BaseVolatilityEstimator(mock_model)

    with pytest.raises(NotFittedError):
        raise vol.compute()

    vol.fit()
    with pytest.raises(ValueError):
        raise vol.compute()

    vol.report = report

    pd.testing.assert_frame_equal(vol.compute(), report)
    pd.testing.assert_frame_equal(vol.compute(metrics=['roc_auc']), report.loc[['roc_auc']])
    pd.testing.assert_frame_equal(vol.compute(metrics='roc_auc'), report.loc[['roc_auc']])


def test_plot(report, mock_model, iterations_train, iterations_test, iterations_delta):
    num_figures_before = plt.gcf().number

    with patch.object(BaseVolatilityEstimator, 'compute', return_value=report.loc[['roc_auc']]) as mock_compute:
        with patch.object(BaseVolatilityEstimator, 'get_samples_to_plot',
                          return_value=(iterations_train, iterations_test, iterations_delta)) as mock_get_samples:

            vol = BaseVolatilityEstimator(mock_model)
            vol.fitted = True

            vol.plot(metrics='roc_auc')
            mock_compute.assert_called_with(metrics='roc_auc')
            mock_get_samples.assert_called_with(metric_name='roc_auc')


    num_figures_after = plt.gcf().number
    assert num_figures_after == num_figures_before + 1


def test_get_samples_to_plot(mock_model, iteration_results, iterations_train, iterations_test, iterations_delta):
    vol = BaseVolatilityEstimator(mock_model)
    vol.fitted = True
    vol.iterations_results=iteration_results

    train, test, delta = vol.get_samples_to_plot(metric_name='roc_auc')
    pd.testing.assert_series_equal(train, iterations_train)
    pd.testing.assert_series_equal(test, iterations_test)
    pd.testing.assert_series_equal(delta, iterations_delta)


def test_create_report(mock_model, iteration_results, report):
    vol = BaseVolatilityEstimator(mock_model)
    vol.fitted = True
    vol.iterations_results = iteration_results

    vol.create_report()
    pd.testing.assert_frame_equal(vol.report, report, check_less_precise=3)


def test_compute_mean_std_from_runs(mock_model, iteration_results):
    vol = BaseVolatilityEstimator(mock_model)
    results = vol.compute_mean_std_from_runs(iteration_results[iteration_results['metric_name'] == 'roc_auc'])
    expected_results = [0.8, 0.08164, 0.7, 0.08164, 0.1, 0]
    for idx, item in enumerate(results):
        assert pytest.approx(item, 0.01) == expected_results[idx]


def test_compute_stats_tests_values(mock_model, iteration_results):
    vol = BaseVolatilityEstimator(mock_model, stats_tests_to_apply=['KS'])

    with patch.object(DistributionStatistics, 'compute', return_value=(0.1, 0.05)):
        stats = vol.compute_stats_tests_values(iteration_results)

    assert stats[0] == 0.1
    assert stats[1] == 0.05


def test_fit_compute(mock_model, report, X_df, y_series):
    vol = BaseVolatilityEstimator(mock_model)

    with patch.object(BaseVolatilityEstimator, 'fit') as mock_fit:
        with patch.object(BaseVolatilityEstimator, 'compute', return_value=report) as mock_compute:
            result = vol.fit_compute(X_df, y_series)

            mock_fit.assert_called_with(X_df, y_series)
            mock_compute.assert_called_with()

    pd.testing.assert_frame_equal(result, report)
