import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probatus.metric_uncertainty.metric import get_metric
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from probatus.utils import assure_numpy_array, NotFittedError, Scorer
from probatus.metric_uncertainty.utils import check_sampling_input, assure_list_of_strings, assure_list_values_allowed
from probatus.stat_tests import DistributionStatistics
import warnings

class BaseVolatilityEstimator(object):
    """
    Base object for estimating volatility estimation.

    Args:
        model: Binary classification model or pipeline
        X: (pd.DataFrame or nparray) Array with samples and features
        y: (pd.DataFrame or nparray) with targets
        metrics : (string, list of strings, Scorer or list of Scorers) metrics for which the score is calculated.
                It can be either a name or list of names of metrics that are supported by Scorer class: 'auc',
                 'accuracy', 'average_precision','neg_log_loss', 'neg_brier_score', 'precision', 'recall', 'jaccard'.
                 In case a custom metric is used, one can create own Scorer (probatus.utils) and provide as a metric.
        test_prc: (Optional float) Percentage of input data used as test. By default 0.25
        n_jobs: (Optional int) number of parallel executions. If -1 use all available cores. By default 1
        compute_stats_tests: (Optional flag) indicates whether the statistical tests should be applied for the
                difference of distribution of metrics on train and test. By default it is True
        stats_tests_to_apply: (Optional string or list of strings) List of tests to apply. Available options are:
                    'ES': Epps-Singleton
                    'KS': Kolmogorov-Smirnov statistic
                    'PSI': Population Stability Index
                    'SW': Shapiro-Wilk based difference statistic
                    'AD': Anderson-Darling TS
                    By default 'KS' and 'SW' tests are applied
        random_state: (Optional int) the seed used by the random number generator
        mean_decimals: (Optional int) number of decimals in the approximation of mean of metric volatility. By default 4
        std_decimals: (Optional int) number of decimals in the approximation of std of metric volatility. By default 4
    """
    def __init__(self, model, X, y, metrics, test_prc=0.25, n_jobs=1, compute_stats_tests=False,
                 stats_tests_to_apply=['KS', 'SW'], random_state=42, mean_decimals=4, std_decimals=4,
                 *args, **kwargs):
        self.model = model
        self.X = assure_numpy_array(X)
        self.y = assure_numpy_array(y)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.test_prc=test_prc
        self.iterations_results = None
        self.report = None
        self.fitted = False
        self.mean_decimals = mean_decimals
        self.std_decimals = std_decimals
        self.compute_stats_tests = compute_stats_tests
        self.allowed_stats_tests = DistributionStatistics.statistical_test_list

        self.stats_tests_to_apply = assure_list_of_strings(stats_tests_to_apply, 'stats_tests_to_apply')
        assure_list_values_allowed(variable=self.stats_tests_to_apply,
                                   variable_name='stats_tests_to_apply',
                                   allowed_values=self.allowed_stats_tests)
        self.stats_tests_objects = []
        if self.compute_stats_tests:
            warnings.warn("Computing statistics for distributions is an experimental feature. While using it, keep in "
                          "mind that the samples of metrics might be correlated.")
            for test_name in self.stats_tests_to_apply:
                self.stats_tests_objects.append(DistributionStatistics(statistical_test=test_name))

        # Append which scorers should be used
        self.scorers = []
        if isinstance(metrics, list):
            for metric in metrics:
                self.append_single_metric_to_scorers(metric)
        else:
            self.append_single_metric_to_scorers(metrics)


    def append_single_metric_to_scorers(self, metric):
        if isinstance(metric, str):
            self.scorers.append(Scorer(metric))
        elif isinstance(metric, Scorer):
            self.scorers.append(metric)
        else:
            raise (ValueError('The metrics should contain either strings'))


    def fit(self, *args, **kwargs):
        """
        Base fit functionality that should be executed before each fit
        """

        # Set seed for results reproducibility
        np.random.seed(self.random_state)

        # Initialize the report and results
        self.iterations_results = None
        self.report = None
        self.fitted = True


    def compute(self, metrics=None, *args, **kwargs):
        """
        Reports the statistics.
        Args:
            metrics: (Optional, str or list of strings) Name or list of names of metrics to be plotted. If not all
            metrics are presented

        Returns:
            (pd Dataframe) Report that contains the evaluation mean and std on train and test sets for each metric
        """

        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))

        if metrics is None:
            return self.report
        else:
            if not isinstance(metrics, list):
                metrics = [metrics]
            return self.report.loc[metrics]

    def plot(self, metrics=None, sampled_distribution=True, bins=10, height_per_subplot=5, width_per_subplot=5):
        """
        Plots distribution of the metric

        Args:
            metrics: (Optional, str or list of strings) Name or list of names of metrics to be plotted. If not all
                metrics are presented
            sampled_distribution: (Optional bool) flag indicating whether the distribution of the bootstrapped data
                should be plotted. If false, the normal distribution based on approximated mean and std is plotted.
                Default True
            bins: (Optional int) Number of bins into which histogram is built
            height_per_subplot: (Optional int) Height of each subplot. Default is 5
            width_per_subplot: (Optional int) Width of each subplot. Default is 5
        """

        target_report = self.compute(metrics=metrics)

        if target_report.shape[0] >= 1:
            print(height_per_subplot*target_report.shape[0])
            print(width_per_subplot*2)
            fig, axs = plt.subplots(target_report.shape[0], 2, figsize=(width_per_subplot*2,
                                                                        height_per_subplot*target_report.shape[0]))

            # Enable traversing the axs
            axs = axs.flatten()
            axis_index = 0

            for metric, row in target_report.iterrows():
                train, test, delta = self.get_samples_to_plot(metric_name=metric,
                                                              sampled_distribution=sampled_distribution)

                axs[axis_index].hist(train, alpha=0.5, label=f'Train {metric}', bins=bins)
                axs[axis_index].hist(test, alpha=0.5, label=f'Test {metric}', bins=bins)
                axs[axis_index].set_title(f'Distributions {metric}')
                axs[axis_index].set(xlabel=f'{metric} score')
                axs[axis_index].legend(loc='upper right')

                axs[axis_index+1].hist(delta, alpha=0.5, label=f'Delta {metric}', bins=bins)
                axs[axis_index+1].set_title(f'Distributions delta {metric}')
                axs[axis_index+1].set(xlabel=f'{metric} scores delta')
                axs[axis_index+1].legend(loc='upper right')

                axis_index+=2

            for ax in axs.flat:
                ax.set(xlabel='Distribution', ylabel='Metric value')

    def get_samples_to_plot(self, metric_name, sampled_distribution=True):
        """
        Selects samples to be plotted

        Args:
            metric_name: (str) Name of metric for which the data should be selected
            sampled_distribution: (bool) flag indicating whether the distribution of the bootstrapped data should be
                plotted. If false, the normal distribution based on approximated mean and std is plotted
        """
        if sampled_distribution:
            current_metric_results = self.iterations_results[self.iterations_results['metric_name'] == metric_name]
            train = current_metric_results['train_score']
            test = current_metric_results['test_score']
            delta = current_metric_results['delta_score']
        else:
            current_metric_distribution = self.report.loc[metric_name]

            train = np.random.normal(current_metric_distribution['train_mean'],
                                     current_metric_distribution['train_std'], 10000)
            test = np.random.normal(current_metric_distribution['test_mean'],
                                    current_metric_distribution['test_std'], 10000)
            delta = np.random.normal(current_metric_distribution['delta_mean'],
                                     current_metric_distribution['delta_std'], 10000)
        return train, test, delta

    def create_report(self):
        """
        Based on the results for each metric for different sampling, mean and std of distributions of all metrics and
        store them as report
        """

        unique_metrics = self.iterations_results['metric_name'].unique()

        # Get columns which will be filled
        stats_tests_columns = []
        for stats_tests_object in self.stats_tests_objects:
            stats_tests_columns.append(f'{stats_tests_object.statistical_test_name} statistic')
            stats_tests_columns.append(f'{stats_tests_object.statistical_test_name} p-value')
        stats_columns = ['train_mean', 'train_std', 'test_mean', 'test_std', 'delta_mean', 'delta_std']
        report_columns = stats_columns + stats_tests_columns

        self.report = pd.DataFrame([], columns=report_columns)

        for metric in unique_metrics:
            metric_iterations_results = self.iterations_results[self.iterations_results['metric_name'] == metric]
            metrics = self.compute_mean_std_from_runs(metric_iterations_results)
            stats_tests_values = self.compute_stats_tests_values(metric_iterations_results)
            metric_row = pd.DataFrame([metrics + stats_tests_values], columns=report_columns, index=[metric])
            self.report = self.report.append(metric_row)


    def compute_mean_std_from_runs(self, metric_iterations_results):
        """
        Compute mean and std of results

        Returns:
            (list of float) List containing mean and std of train, test and deltas
        """
        train_mean_score = np.round(np.mean(metric_iterations_results['train_score']), self.mean_decimals)
        test_mean_score = np.round(np.mean(metric_iterations_results['test_score']), self.mean_decimals)
        delta_mean_score = np.round(np.mean(metric_iterations_results['delta_score']), self.mean_decimals)
        train_std_score = np.round(np.std(metric_iterations_results['train_score']), self.std_decimals)
        test_std_score = np.round(np.std(metric_iterations_results['test_score']), self.std_decimals)
        delta_std_score = np.round(np.std(metric_iterations_results['delta_score']), self.std_decimals)
        return [train_mean_score, train_std_score, test_mean_score, test_std_score, delta_mean_score, delta_std_score]

    def compute_stats_tests_values(self, metric_iterations_results):
        """
        Compute statistics and p-values of specified tests

        Returns:
            (list of float) List containing statistics and p-values of distributions
        """
        statistics = []
        for stats_test in self.stats_tests_objects:
            stats, p_value = \
                stats_test.compute(metric_iterations_results['test_score'], metric_iterations_results['train_score'])
            statistics += [stats, p_value]
        return statistics

    def fit_compute(self,  *args, **kwargs):
        """
        Runs trains and evaluates a number of models on train and test sets extracted using different random seeds.
        Reports the statistics of the selected metric.

        Returns:
            (pd Dataframe) Report that contains the evaluation mean and std on train and test sets for each metric
        """

        self.fit(*args, **kwargs)
        return self.compute(*args, **kwargs)


class LocalVolatilityEstimator(BaseVolatilityEstimator):
    """
    Estimation of volatility of metrics. The estimation is done by splitting the data into train and test multiple times
     and training and scoring a model based on these metrics. THe class allows for choosing whether at each iteration
     the train test split should be the same or different, whether and how the train and test sets should be sampled.

    Args:
        model: Binary classification model or pipeline
        X: (pd.DataFrame or nparray) Array with samples and features
        y: (pd.DataFrame or nparray) with targets
        metrics : (string, list of strings, Scorer or list of Scorers) metrics for which the score is calculated.
                It can be either a name or list of names of metrics that are supported by Scorer class: 'auc',
                 'accuracy', 'average_precision','neg_log_loss', 'neg_brier_score', 'precision', 'recall', 'jaccard'.
                 In case a custom metric is used, one can create own Scorer (probatus.utils) and provide as a metric.
        train_test_split_seed: (Optional int) the seed used for all train test splits if sample_train_test_split_seed is
                set to False. The default value of this parameter is 42
        sample_train_type: (Optional str) string indicating what type of sampling should be applied on train set:
                - None indicates that no additional sampling is done after splitting data
                - 'bootstrap' indicates that sampling with replacement will be performed on train data
                - 'subsample': indicates that sampling without repetition will be performed  on train data
        sample_test_type: (Optional str) string indicating what type of sampling should be applied on test set:
                - None indicates that no additional sampling is done after splitting data
                - 'bootstrap' indicates that sampling with replacement will be performed on test data
                - 'subsample': indicates that sampling without repetition will be performed  on test data
        sample_train_fraction: (Optional float): fraction of train data sampled, if sample_train_type is not None.
                Default value is 1
        sample_test_fraction: (Optional float): fraction of test data sampled, if sample_test_type is not None.
                Default value is 1
        sample_train_test_split_seed: (Optional bool) Flag indicating whether each train test split should be done
                randomly or measurement should be done for single split. Default is True, which indicates that each
                iteration is performed on a random train test split. If the value is False, the random_seed for the
                split is set to train_test_split_seed
        iterations: (Optional int) Number of iterations in seed bootstrapping. By default 1000.
        test_prc: (Optional float) Percentage of input data used as test. By default 0.25
        n_jobs: (Optional int) number of parallel executions. If -1 use all available cores. By default 1
        compute_stats_tests: (Optional flag) indicates whether the statistical tests should be applied for the
                difference of distribution of metrics on train and test. By default it is True
        stats_tests_to_apply: (Optional string or list of strings) List of tests to apply. Available options are:
                    'ES': Epps-Singleton
                    'KS': Kolmogorov-Smirnov statistic
                    'PSI': Population Stability Index
                    'SW': Shapiro-Wilk based difference statistic
                    'AD': Anderson-Darling TS
                    By default 'KS' and 'SW' tests are applied
        random_state: (Optional int) the seed used by the random number generator
        mean_decimals: (Optional int) number of decimals in the approximation of mean of metric volatility. By default 4
        std_decimals: (Optional int) number of decimals in the approximation of std of metric volatility. By default 4
    """

    def __init__(self, model, X, y, metrics, train_test_split_seed=42, sample_train_type=None, sample_test_type=None,
                 sample_train_fraction=1, sample_test_fraction=1, sample_train_test_split_seed=True, iterations=1000,
                  *args, **kwargs):
        super().__init__(model=model, X=X, y=y, metrics=metrics, *args, **kwargs)
        self.iterations = iterations
        self.train_test_split_seed = train_test_split_seed
        self.sample_train_type = sample_train_type
        self.sample_test_type = sample_test_type
        self.sample_train_test_split_seed=sample_train_test_split_seed
        self.sample_train_fraction = sample_train_fraction
        self.sample_test_fraction = sample_test_fraction

        check_sampling_input(sample_train_type, sample_train_fraction, 'train')
        check_sampling_input(sample_test_type, sample_test_fraction, 'test')


    def fit(self):
        """
        Bootstraps a number of random seeds, then splits the data based on the sampled seeds and estimates performance
        of the model based on the split data.
        """
        super().fit()

        if self.sample_train_test_split_seed:
            random_seeds = np.random.random_integers(0, 999999, self.iterations)
        else:
            random_seeds = (np.ones(self.iterations) * self.train_test_split_seed).astype(int)

        results_per_iteration = Parallel(n_jobs=self.n_jobs)(delayed(get_metric)(
            X=self.X, y=self.y, model=self.model, test_size=self.test_prc, split_seed=split_seed,
            scorers=self.scorers, sample_train_type=self.sample_train_type, sample_test_type=self.sample_test_type,
            sample_train_fraction=self.sample_train_fraction, sample_test_fraction=self.sample_test_fraction
        ) for split_seed in tqdm(random_seeds))

        self.iterations_results = pd.concat(results_per_iteration, ignore_index=True)

        self.create_report()
