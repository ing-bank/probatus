# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probatus.metric_volatility.metric import get_metric
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from probatus.utils import (
    get_scorers,
    assure_list_of_strings,
    assure_list_values_allowed,
    BaseFitComputePlotClass,
    preprocess_data,
    preprocess_labels,
)
from probatus.metric_volatility.utils import check_sampling_input
from probatus.stat_tests import DistributionStatistics
import warnings


class BaseVolatilityEstimator(BaseFitComputePlotClass):
    """
    Base object for estimating volatility estimation.

    This class is a base class, therefore cannot be used on its
    own. Implements common API that can be used by all subclasses.
    """

    def __init__(
        self,
        clf,
        scoring="roc_auc",
        test_prc=0.25,
        n_jobs=1,
        stats_tests_to_apply=None,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers, optional):
                Metrics for which the score is calculated. It can be either a name or list of names metric names and
                needs to be aligned with predefined classification scorers names in sklearn
                 ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
                Another option is using probatus.utils.Scorer to define a custom metric.

            test_prc (float, optional):
                Percentage of input data used as test. By default 0.25.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            stats_tests_to_apply (None, string or list of strings, optional):
                List of tests to apply. Available options:

                - `'ES'`: Epps-Singleton,
                - `'KS'`: Kolmogorov-Smirnov statistic,
                - `'PSI'`: Population Stability Index,
                - `'SW'`: Shapiro-Wilk based difference statistic,
                - `'AD'`: Anderson-Darling TS.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings and indication of progress in fitting the object.
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """
        self.clf = clf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.test_prc = test_prc
        self.iterations_results = None
        self.report = None
        self.verbose = verbose
        self.allowed_stats_tests = list(DistributionStatistics.statistical_test_dict.keys())

        # TODO set reasonable default value for the parameter, to choose the statistical test for the user for different
        #  ways to compute volatility
        if stats_tests_to_apply is not None:
            self.stats_tests_to_apply = assure_list_of_strings(stats_tests_to_apply, "stats_tests_to_apply")
            assure_list_values_allowed(
                variable=self.stats_tests_to_apply,
                variable_name="stats_tests_to_apply",
                allowed_values=self.allowed_stats_tests,
            )
        else:
            self.stats_tests_to_apply = []

        self.stats_tests_objects = []
        if len(self.stats_tests_to_apply) > 0:
            if self.verbose > 0:
                warnings.warn(
                    "Computing statistics for distributions is an experimental feature. While using it, keep "
                    "in mind that the samples of metrics might be correlated."
                )
            for test_name in self.stats_tests_to_apply:
                self.stats_tests_objects.append(DistributionStatistics(statistical_test=test_name))

        self.scorers = get_scorers(scoring)

    def fit(self, *args, **kwargs):
        """
        Base fit functionality that should be executed before each fit.

        Returns:
            (BaseVolatilityEstimator):
                Fitted object.
        """
        # Set seed for results reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize the report and results
        self.iterations_results = None
        self.report = None
        self.fitted = True
        return self

    def compute(self, metrics=None):
        """
        Reports the statistics.

        Args:
            metrics (str or list of strings, optional):
                Name or list of names of metrics to be plotted. If not all metrics are presented.

        Returns:
            (pandas.Dataframe):
                Report that contains the evaluation mean and std on train and test sets for each metric.
        """
        self._check_if_fitted()
        if self.report is None:
            raise (
                ValueError(
                    "Report is None, thus it has not been computed by fit method. Please extend the "
                    "BaseVolatilityEstimator class, overwrite fit method, and within fit run compute_report()"
                )
            )

        if metrics is None:
            return self.report
        else:
            if not isinstance(metrics, list):
                metrics = [metrics]
            return self.report.loc[metrics]

    def plot(
        self,
        metrics=None,
        bins=10,
        show=True,
        height_per_subplot=5,
        width_per_subplot=5,
    ):
        """
        Plots distribution of the metric.

        Args:
            metrics (str or list of strings, optional):
                Name or list of names of metrics to be plotted. If not all metrics are presented.

            bins (int, optional):
                Number of bins into which histogram is built.

            show (bool, optional):
                If True, the plots are showed to the user, otherwise they are not shown. Not showing plot can be useful,
                when you want to edit the returned axis, before showing it.

            height_per_subplot (int, optional):
                Height of each subplot. Default is 5.

            width_per_subplot (int, optional):
                Width of each subplot. Default is 5.

        Returns
            (list(matplotlib.axes)):
                Axes that include the plot.
        """

        target_report = self.compute(metrics=metrics)

        if target_report.shape[0] >= 1:
            fig, axs = plt.subplots(
                target_report.shape[0],
                2,
                figsize=(
                    width_per_subplot * 2,
                    height_per_subplot * target_report.shape[0],
                ),
            )

            # Enable traversing the axs
            axs = axs.flatten()
            axis_index = 0

            for metric, row in target_report.iterrows():
                train, test, delta = self._get_samples_to_plot(metric_name=metric)

                axs[axis_index].hist(train, alpha=0.5, label="Train {}".format(metric), bins=bins)
                axs[axis_index].hist(test, alpha=0.5, label="Test {}".format(metric), bins=bins)
                axs[axis_index].set_title("Distributions {}".format(metric))
                axs[axis_index].legend(loc="upper right")

                axs[axis_index + 1].hist(delta, alpha=0.5, label="Delta {}".format(metric), bins=bins)
                axs[axis_index + 1].set_title("Distributions delta {}".format(metric))
                axs[axis_index + 1].legend(loc="upper right")

                axis_index += 2

            for ax in axs.flat:
                ax.set(xlabel="{} score".format(metric), ylabel="Results count")

            if show:
                plt.show()
            else:
                plt.close()

            return axs

    def _get_samples_to_plot(self, metric_name):
        """
        Selects samples to be plotted.

        Args:
            metric_name (str):
                Name of metric for which the data should be selected.
        """
        current_metric_results = self.iterations_results[self.iterations_results["metric_name"] == metric_name]
        train = current_metric_results["train_score"]
        test = current_metric_results["test_score"]
        delta = current_metric_results["delta_score"]

        return train, test, delta

    def _create_report(self):
        """
        Create a report.

        Based on the results for each metric for different sampling, mean and std of distributions of all metrics and
        store them as report.
        """
        unique_metrics = self.iterations_results["metric_name"].unique()

        # Get columns which will be filled
        stats_tests_columns = []
        for stats_tests_object in self.stats_tests_objects:
            stats_tests_columns.append("{} statistic".format(stats_tests_object.statistical_test_name))
            stats_tests_columns.append("{} p-value".format(stats_tests_object.statistical_test_name))
        stats_columns = [
            "train_mean",
            "train_std",
            "test_mean",
            "test_std",
            "delta_mean",
            "delta_std",
        ]
        report_columns = stats_columns + stats_tests_columns

        self.report = pd.DataFrame([], columns=report_columns)

        for metric in unique_metrics:
            metric_iterations_results = self.iterations_results[self.iterations_results["metric_name"] == metric]
            metrics = self._compute_mean_std_from_runs(metric_iterations_results)
            stats_tests_values = self._compute_stats_tests_values(metric_iterations_results)
            metric_row = pd.DataFrame([metrics + stats_tests_values], columns=report_columns, index=[metric])
            self.report = self.report.append(metric_row)

    def _compute_mean_std_from_runs(self, metric_iterations_results):
        """
        Compute mean and std of results.

        Args:
            metric_iterations_results (pandas.DataFrame):
                Scores for a single metric for each iteration.

        Returns:
            (list):
                List containing mean and std of train, test and deltas.
        """
        train_mean_score = np.mean(metric_iterations_results["train_score"])
        test_mean_score = np.mean(metric_iterations_results["test_score"])
        delta_mean_score = np.mean(metric_iterations_results["delta_score"])
        train_std_score = np.std(metric_iterations_results["train_score"])
        test_std_score = np.std(metric_iterations_results["test_score"])
        delta_std_score = np.std(metric_iterations_results["delta_score"])
        return [
            train_mean_score,
            train_std_score,
            test_mean_score,
            test_std_score,
            delta_mean_score,
            delta_std_score,
        ]

    def _compute_stats_tests_values(self, metric_iterations_results):
        """
        Compute statistics and p-values of specified tests.

        Args:
            metric_iterations_results (pandas.DataFrame):
                Scores for a single metric for each iteration.

        Returns:
            (list):
                List containing statistics and p-values of distributions.
        """
        statistics = []
        for stats_test in self.stats_tests_objects:
            stats, p_value = stats_test.compute(
                metric_iterations_results["test_score"],
                metric_iterations_results["train_score"],
            )
            statistics += [stats, p_value]
        return statistics

    def fit_compute(self, *args, **kwargs):
        """
        Fit compute.

        Runs trains and evaluates a number of models on train and test sets extracted using different random seeds.
            Reports the statistics of the selected metric.

        Takes as arguments the same parameters as fit() method.

        Returns:
            (pandas.Dataframe):
                Report that contains the evaluation mean and std on train and test sets for each metric.
        """
        self.fit(*args, **kwargs)
        return self.compute()


class TrainTestVolatility(BaseVolatilityEstimator):
    """
    Estimation of volatility of metrics.

    The estimation is done by splitting the data into train and test multiple times
        and training and scoring a model based on these metrics. The class allows for choosing whether at each iteration
        the train test split should be the same or different, whether and how the train and test sets should be sampled.

    Examples:

    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.metric_volatility import TrainTestVolatility
    X, y = make_classification(n_features=4)
    clf = RandomForestClassifier()
    volatility = TrainTestVolatility(clf, iterations=10 , test_prc = 0.5)
    volatility_report = volatility.fit_compute(X, y)
    volatility.plot()
    ```

    <img src="../img/metric_volatility_train_test.png" width="500" />
    """

    def __init__(
        self,
        clf,
        iterations=1000,
        scoring="roc_auc",
        sample_train_test_split_seed=True,
        train_sampling_type=None,
        test_sampling_type=None,
        train_sampling_fraction=1,
        test_sampling_fraction=1,
        test_prc=0.25,
        n_jobs=1,
        stats_tests_to_apply=None,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            iterations (int, optional):
                Number of iterations in seed bootstrapping. By default 1000.

            scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers, optional):
                Metrics for which the score is calculated. It can be either a name or list of names metric names and
                needs to be aligned with predefined classification scorers names in sklearn
                ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
                Another option is using probatus.utils.Scorer to define a custom metric.

            sample_train_test_split_seed (bool, optional):
                Flag indicating whether each train test split should be done
                randomly or measurement should be done for single split. Default is True, which indicates that each.
                iteration is performed on a random train test split. If the value is False, the random_seed for the
                split is set to train_test_split_seed.

            train_sampling_type (str, optional):
                String indicating what type of sampling should be applied on train set:

                - `None` indicates that no additional sampling is done after splitting data,
                - `'bootstrap'` indicates that sampling with replacement will be performed on train data,
                - `'subsample'` indicates that sampling without repetition will be performed  on train data.

            test_sampling_type (str, optional):
                String indicating what type of sampling should be applied on test set:

                - `None` indicates that no additional sampling is done after splitting data,
                - `'bootstrap'` indicates that sampling with replacement will be performed on test data,
                - `'subsample'` indicates that sampling without repetition will be performed  on test data.

            train_sampling_fraction (float, optional):
                Fraction of train data sampled, if sample_train_type is not None.
                Default value is 1.

            test_sampling_fraction (float, optional):
                Fraction of test data sampled, if sample_test_type is not None. Default value is 1.

            test_prc (float, optional):
                Percentage of input data used as test. By default 0.25.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            stats_tests_to_apply (None, string or list of strings, optional):
                List of tests to apply, default is None. Available options:

                - `'ES'`: Epps-Singleton,
                - `'KS'`: Kolmogorov-Smirnov statistic,
                - `'PSI'`: Population Stability Index,
                - `'SW'`: Shapiro-Wilk based difference statistic,
                - `'AD'`: Anderson-Darling TS.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """
        super().__init__(
            clf=clf,
            scoring=scoring,
            test_prc=test_prc,
            n_jobs=n_jobs,
            stats_tests_to_apply=stats_tests_to_apply,
            verbose=verbose,
            random_state=random_state,
        )
        self.iterations = iterations
        self.train_sampling_type = train_sampling_type
        self.test_sampling_type = test_sampling_type
        self.sample_train_test_split_seed = sample_train_test_split_seed
        self.train_sampling_fraction = train_sampling_fraction
        self.test_sampling_fraction = test_sampling_fraction

        check_sampling_input(train_sampling_type, train_sampling_fraction, "train")
        check_sampling_input(test_sampling_type, test_sampling_fraction, "test")

    def fit(self, X, y, column_names=None):
        """
        Fit.

        Bootstraps a number of random seeds, then splits the data based on the sampled seeds and estimates performance
            of the model based on the split data.

        Args:
            X (pandas.DataFrame or numpy.ndarray):
                Array with samples and features.

            y (pandas.Series or numpy.ndarray):
                Array with targets.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

        Returns:
            (TrainTestVolatility):
                Fitted object.
        """
        super().fit()

        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, y_name="y", index=self.X.index, verbose=self.verbose)

        if self.sample_train_test_split_seed:
            random_seeds = np.random.random_integers(0, 999999, self.iterations)
        else:
            random_seeds = (np.ones(self.iterations)).astype(int)
            if self.random_state:
                random_seeds = random_seeds * self.random_state

        if self.verbose > 0:
            random_seeds = tqdm(random_seeds)

        results_per_iteration = Parallel(n_jobs=self.n_jobs)(
            delayed(get_metric)(
                X=self.X,
                y=self.y,
                clf=self.clf,
                test_size=self.test_prc,
                split_seed=split_seed,
                scorers=self.scorers,
                train_sampling_type=self.train_sampling_type,
                test_sampling_type=self.test_sampling_type,
                train_sampling_fraction=self.train_sampling_fraction,
                test_sampling_fraction=self.test_sampling_fraction,
            )
            for split_seed in random_seeds
        )

        self.iterations_results = pd.concat(results_per_iteration, ignore_index=True)

        self._create_report()
        return self


class SplitSeedVolatility(TrainTestVolatility):
    """
    Estimation of volatility of metrics depending on the seed used to split the data.

    At every iteration it splits the
        data into train and test set using a different stratified split and volatility of the metrics is calculated.

    Examples:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.metric_volatility import SplitSeedVolatility
    X, y = make_classification(n_features=4)
    clf = RandomForestClassifier()
    volatility = SplitSeedVolatility(clf, iterations=10 , test_prc = 0.5)
    volatility_report = volatility.fit_compute(X, y)
    volatility.plot()
    ```
    <img src="../img/metric_volatility_split_seed.png" width="500" />

    """

    def __init__(
        self,
        clf,
        iterations=1000,
        scoring="roc_auc",
        test_prc=0.25,
        n_jobs=1,
        stats_tests_to_apply=None,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            iterations (int, optional):
                Number of iterations in seed bootstrapping. By default 1000.

            scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers, optional):
                Metrics for which the score is calculated. It can be either a name or list of names metric names and
                needs to be aligned with predefined classification scorers names in sklearn
                ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
                Another option is using probatus.utils.Scorer to define a custom metric.

            test_prc (float, optional):
                Percentage of input data used as test. By default 0.25.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            stats_tests_to_apply (None, string or list of strings, optional):
                List of tests to apply, default is None. Available options:

                - `'ES'`: Epps-Singleton,
                - `'KS'`: Kolmogorov-Smirnov statistic,
                - `'PSI'`: Population Stability Index,
                - `'SW'`: Shapiro-Wilk based difference statistic,
                - `'AD'`: Anderson-Darling TS.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """
        super().__init__(
            clf=clf,
            sample_train_test_split_seed=True,
            train_sampling_type=None,
            test_sampling_type=None,
            train_sampling_fraction=1,
            test_sampling_fraction=1,
            iterations=iterations,
            scoring=scoring,
            test_prc=test_prc,
            n_jobs=n_jobs,
            stats_tests_to_apply=stats_tests_to_apply,
            verbose=verbose,
            random_state=random_state,
        )


class BootstrappedVolatility(TrainTestVolatility):
    """
    Estimation of volatility of metrics by bootstrapping both train and test set.

    By default at every iteration the
        train test split is the same. The test shows volatility of metric with regards to sampling different rows from
        static train and test sets.

    Examples:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.metric_volatility import BootstrappedVolatility
    X, y = make_classification(n_features=4)
    clf = RandomForestClassifier()
    volatility = BootstrappedVolatility(clf, iterations=10 , test_prc = 0.5)
    volatility_report = volatility.fit_compute(X, y)
    volatility.plot()
    ```
    <img src="../img/metric_volatility_bootstrapped.png" width="500" />
    """

    def __init__(
        self,
        clf,
        iterations=1000,
        scoring="roc_auc",
        train_sampling_fraction=1,
        test_sampling_fraction=1,
        test_prc=0.25,
        n_jobs=1,
        stats_tests_to_apply=None,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            iterations (int, optional):
                Number of iterations in seed bootstrapping. By default 1000.

            scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers, optional):
                Metrics for which the score is calculated. It can be either a name or list of names metric names and
                needs to be aligned with predefined classification scorers names in sklearn
                ([link](https://scikit-learn.org/stable/modules/model_evaluation.html)).
                Another option is using probatus.utils.Scorer to define a custom metric.

            train_sampling_fraction (float, optional):
                Fraction of train data sampled, if sample_train_type is not None. Default value is 1.

            test_sampling_fraction (float, optional):
                Fraction of test data sampled, if sample_test_type is not None. Default value is 1.

            test_prc (float, optional):
                Percentage of input data used as test. By default 0.25.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            stats_tests_to_apply (None, string or list of strings, optional):
                List of tests to apply, default is None. Available options:

                - `'ES'`: Epps-Singleton,
                - `'KS'`: Kolmogorov-Smirnov statistic,
                - `'PSI'`: Population Stability Index,
                - `'SW'`: Shapiro-Wilk based difference statistic,
                - `'AD'`: Anderson-Darling TS.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """
        super().__init__(
            clf=clf,
            sample_train_test_split_seed=False,
            train_sampling_type="bootstrap",
            test_sampling_type="bootstrap",
            iterations=iterations,
            scoring=scoring,
            train_sampling_fraction=train_sampling_fraction,
            test_sampling_fraction=test_sampling_fraction,
            test_prc=test_prc,
            n_jobs=n_jobs,
            stats_tests_to_apply=stats_tests_to_apply,
            verbose=verbose,
            random_state=random_state,
        )
