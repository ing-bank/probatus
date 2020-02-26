import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probatus.metric_uncertainty.metric import get_metric
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from probatus.utils import assure_numpy_array, NotFittedError, Scorer
from probatus.metric_uncertainty import delong
from probatus.metric_uncertainty.utils import max_folds
from probatus.metric_uncertainty.sampling import stratified_random
from probatus.metric_uncertainty.utils import slicer
from probatus.metric_uncertainty.metric import get_metric_folds


class BaseVolatilityEstimator(object):
    """
    Base object for estimating volatility estimation.

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        metrics : (string, list of strings, Scorer object or list of Scorer objects) metrics for which the score is
                calculated. It can be either a name or list of names of metrics that are supported by Scorer class.
                It can also be a single Scorer object of list of these objects.
        test_prc: (Optional float) Percentage of input data used as test. By default 0.25
        n_jobs: (Optional int) number of parallel executions. If -1 use all available cores. By default 1
        random_state: (Optional int) the seed used by the random number generator
        mean_decimals: (Optional int) number of decimals in the approximation of mean of metric volatility. By default 4
        std_decimals: (Optional int) number of decimals in the approximation of std of metric volatility. By default 4
    """
    def __init__(self, model, X, y, metrics, test_prc=0.25, n_jobs=1, random_state=42, mean_decimals=4, std_decimals=4,
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
            raise (ValueError('The metrics should contain either strings or Scorer objects'))


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

    def plot(self, metrics=None, sampled_distribution=True):
        """
        Plots distribution of the metric

        Args:
            metrics: (Optional, str or list of strings) Name or list of names of metrics to be plotted. If not all
                metrics are presented
            sampled_distribution: (bool) flag indicating whether the distribution of the bootstrapped data should be
                plotted. If false, the normal distribution based on approximated mean and std is plotted
        """

        target_report = self.compute(metrics=metrics)

        for metric, row in target_report.iterrows():
            train, test, delta = self.get_samples_to_plot(metric_name=metric, sampled_distribution=sampled_distribution)

            plt.hist(train, alpha=0.5, label=f'Train {metric}')
            plt.hist(test, alpha=0.5, label=f'Test {metric}')
            plt.title(f'Distributions of train and test {metric}')
            plt.xlabel(f'{metric} score')
            plt.ylabel(f'Frequency')
            plt.legend(loc='upper right')
            plt.show()

            plt.hist(delta, alpha=0.5, label=f'Delta {metric}')
            plt.title(f'Distribution of train test {metric} differences')
            plt.xlabel(f'{metric} scores delta')
            plt.ylabel(f'Frequency')
            plt.legend(loc='upper right')
            plt.show()

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

        report_columns = ['train_mean', 'train_std', 'test_mean', 'test_std', 'delta_mean', 'delta_std']
        self.report = pd.DataFrame([], columns=report_columns)

        for metric in unique_metrics:
            metric_iterations_results = self.iterations_results[self.iterations_results['metric_name'] == metric]
            metric_row = pd.DataFrame(self.compute_mean_std_from_runs(metric_iterations_results),
                                      columns=report_columns, index=[metric])
            self.report = self.report.append(metric_row)

    def compute_mean_std_from_runs(self, metric_iterations_results):
        """
        Compute mean and std of results

        Returns:
            (pd.Dataframe) Report that contains single row of results for the desired metric statistics
        """
        train_mean_score = np.round(np.mean(metric_iterations_results['train_score']), self.mean_decimals)
        test_mean_score = np.round(np.mean(metric_iterations_results['test_score']), self.mean_decimals)
        delta_mean_score = np.round(np.mean(metric_iterations_results['delta_score']), self.mean_decimals)
        train_std_score = np.round(np.std(metric_iterations_results['train_score']), self.std_decimals)
        test_std_score = np.round(np.std(metric_iterations_results['test_score']), self.std_decimals)
        delta_std_score = np.round(np.std(metric_iterations_results['delta_score']), self.std_decimals)
        return [[train_mean_score, train_std_score, test_mean_score, test_std_score, delta_mean_score, delta_std_score]]

    def fit_compute(self,  *args, **kwargs):
        """
        Runs trains and evaluates a number of models on train and test sets extracted using different random seeds.
        Reports the statistics of the selected metric.

        Returns:
            (pd Dataframe) Report that contains the evaluation mean and std on train and test sets for each metric
        """

        self.fit(*args, **kwargs)
        return self.compute(*args, **kwargs)


class BootstrapSeedVolatility(BaseVolatilityEstimator):
    """
    Estimation of volatility of metrics. The estimation is done by splitting the data into train and test multiple times
    using bootstrapped random seed. Then the model is trained and scored on these datasets.

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        metrics : (string, list of strings, Scorer object or list of Scorer objects) metrics for which the score is
                calculated. It can be either a name or list of names of metrics that are supported by Scorer class.
                It can also be a single Scorer object of list of these objects.
        iterations: (Optional int) Number of iterations in seed bootstrapping. By default 1000.
        test_prc: (Optional float) Percentage of input data used as test. By default 0.25
        n_jobs: (Optional int) number of parallel executions. If -1 use all available cores. By default 1
        random_state: (Optional int) the seed used by the random number generator
        mean_decimals: (Optional int) number of decimals in the approximation of mean of metric volatility. By default 4
        std_decimals: (Optional int) number of decimals in the approximation of std of metric volatility. By default 4
    """
    def __init__(self, model, X, y, metrics, iterations=1000, *args, **kwargs):
        super().__init__(model=model, X=X, y=y, metrics=metrics, *args, **kwargs)
        self.iterations = iterations

    def fit(self):
        """
        Bootstraps a number of random seeds, then splits the data based on the sampled seeds and estimates performance
        of the model based on the split data.
        """
        super().fit()

        random_seeds = np.random.random_integers(0, 999999, self.iterations)

        results_per_iteration = Parallel(n_jobs=self.n_jobs)(delayed(get_metric)(
            self.X, self.y, self.model, self.test_prc, seed, self.scorers) for seed in tqdm(random_seeds))

        self.iterations_results = pd.concat(results_per_iteration, ignore_index=True)

        self.create_report()

