import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probatus.metric_uncertainty.metric import get_metric
from joblib import Parallel, delayed
from tqdm import tqdm
from probatus.utils import assure_numpy_array
from probatus.metric_uncertainty import delong
from probatus.metric_uncertainty.utils import max_folds
from probatus.metric_uncertainty.sampling import stratified_random
from probatus.metric_uncertainty.utils import slicer
from probatus.metric_uncertainty.metric import get_metric_folds


class VolatilityEstimation(object):
    """
    Draws N random samples from the data to create new train/test splits and calculate metric of interest.
    After collecting multiple metrics per split, summary statistics of the metric are reported.
    This provides uncertainty levels around the metric if the train/test data is resampled.  

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        evaluator : dict with name of the metric as a key and array holding [scoring function, scoring type]
            e.g. {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
        n_jobs: number of cores to be used. -1 sets it to use all the free cores
        method: str type of estimation which should be used, currently supports:
                boot_global - bootstrap replicates with global estimation of the AUC uncertainty
                with non onverlapping resampling
                boot_seed - boostrap replicates with local estimation of the AUC uncertainty (fixed split)
                and overlapping resampling
        random_state: the seed used by the random number generator

    """
    def __init__(self, model, X, y, evaluator, method, n_jobs=1, random_state=42):
        self.model = model
        self.X = assure_numpy_array(X)
        self.y = assure_numpy_array(y)
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.metrics_dict = {}
        self.results_df = pd.DataFrame()
        self.method = method
        self.random_state = random_state

    def fit(self, test_prc, iterations=1000):
        """
        Runs trains and evaluates a number of models on train and test sets extracted using different random seeds.
        The calculated target metrics are stored inside the object as metrics_list.
        Statistics based on these metrics can be computed using compute().

        Args:
            test_prc: (float) percentage of data used for test partition as hold out
            iterations: (int) number of iterations of model training

        Returns:
            Dictionary with metrics data from sampling
        """

        # Reproducable results
        np.random.seed(self.random_state)
        
        metrics = list(self.evaluator.keys())

        for evaluator_i in metrics:

            if self.method == 'boot_seed':
                random_seeds = np.random.random_integers(0, 999999, iterations)
                results = Parallel(n_jobs=self.n_jobs)(delayed(get_metric)(self.X, 
                                                                           self.y, 
                                                                           self.model, 
                                                                           test_prc, 
                                                                           i, 
                                                                           self.evaluator[evaluator_i][0],
                                                                           self.evaluator[evaluator_i][1])
                                                       for i in tqdm(random_seeds))

                self.metrics_dict[evaluator_i] = np.array(results)
            
            elif self.method == 'boot_global':

                results = []
                X_train, X_test, y_train, y_test = stratified_random(self.X, self.y, test_prc)
                top_k = max_folds(y_train) 
                if top_k > 11:
                    top_k = 11
                for k in tqdm(range(1, top_k + 1), position = 0):

                    if k == 1:
                        x_slice = [X_train]
                        y_slice = [y_train]
                    
                    else:
                        x_slice, y_slice = slicer(X_train, y_train, k)

                    results_i = get_metric_folds(x_slice, 
                                                 y_slice, 
                                                 self.model, 
                                                 X_test, 
                                                 y_test,
                                                 self.evaluator[evaluator_i][0],
                                                 self.evaluator[evaluator_i][1])
                    results.append(results_i)
                self.metrics_dict[evaluator_i] = np.array(results) 

            elif self.method == 'delong':    
                X_train, X_test, y_train, y_test = stratified_random(self.X, self.y, test_prc)
                model = self.model.fit(X_train, y_train)

                y_pred_test = model.predict_proba(X_test)[:, 1]
                y_pred_train = model.predict_proba(X_train)[:, 1]

                auc_train, auc_cov_train = delong.delong_roc_variance(y_train, y_pred_train)
                auc_test, auc_cov_test = delong.delong_roc_variance(y_test, y_pred_test)

                self.metrics_dict[evaluator_i] = np.array([auc_train, auc_test,
                                                           auc_train - auc_test,
                                                           auc_cov_train,
                                                           auc_cov_test,
                                                           0]).reshape(1, 6)

            metric_data = self.metrics_dict[evaluator_i]

            self.results_df = self.create_results_df(metric_data, evaluator_i, self.method)

    def compute(self, metric, mean_decimals=4, std_decimals=4):
        """
        Reports the statistics of the selected metric.

        Args:
            metric: (str) name of the metric to report
            mean_decimals: (int) number of decimals in approximation of the mean
            std_decimals: (int) number of decimals in approximation of the std

        Returns:
            (pd Dataframe) Report that contains the evaluation mean and std on train and test sets.
        """

        # TODO Make a check if a metric not in already computed metrics
        results = self.results_df.loc[[metric]]

        # TODO allow for computations of multiple metrics at the same time
        for column_name in results.columns:
            if 'mean_' in column_name:
                results[column_name] = results[column_name].round(mean_decimals)
            else:
                results[column_name] = results[column_name].round(std_decimals)

        return results

    def plot(self, metric):
        """
        Plots distribution of the metric

        Args:
            metric: str name of the metric to report

        """  

        metric_data = self.metrics_dict[metric]
        results = self.results_df.loc[[metric]]

        if self.method == 'boot_seed':
            train = metric_data[:, 0]
            test = metric_data[:, 1]
            delta = metric_data[:, 2]
        elif self.method == 'boot_global' or self.method == 'delong':
            train = np.random.normal(results['mean_train'], results['std_train'], 10000)
            test = np.random.normal(results['mean_test'], results['std_test'], 10000)
            delta = np.random.normal(results['mean_delta'], results['std_delta'], 10000)

        plt.hist(train, alpha=0.5, label='Train')
        plt.hist(test, alpha=0.5, label='Test')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(delta, alpha=0.5, label='Delta')
        plt.legend(loc='upper right')
        plt.show()

    @staticmethod
    def create_results_df(data, metric, method):
        """
        Creates a dataframe using statistics related to metrics

        Args:
            data: name of the variable that keeps the statisics
            metric: metric used as index
            method: type of estimation which is used

        Returns:
            (pd.Dataframe) Report that contains the desired statistics statistics

        """

        results = pd.DataFrame()

        results.loc[metric, 'mean_train'] = np.mean(data[:, 0])
        results.loc[metric, 'mean_test'] = np.mean(data[:, 1])
        results.loc[metric, 'mean_delta'] = np.mean(data[:, 2])

        if method == 'boot_seed':
            results.loc[metric, 'std_train'] = np.std(data[:, 0])
            results.loc[metric, 'std_test'] = np.std(data[:, 1])
            results.loc[metric, 'std_delta'] = np.std(data[:, 2])
        elif method == 'boot_global' or method == 'delong':
            # Here in corresponding parts of the 'data', variances are kept.
            # Therefore we take their average first and then convert to std.
            results.loc[metric, 'std_train'] = np.sqrt(np.mean(data[:, 3]))
            results.loc[metric, 'std_test'] = np.sqrt(np.mean(data[:, 4]))
            results.loc[metric, 'std_delta'] = np.sqrt(np.mean(data[:, 5]))

        return results

    def fit_compute(self, test_prc, metric, iterations=1000, mean_decimals=4, std_decimals=4):
        """
        Runs trains and evaluates a number of models on train and test sets extracted using different random seeds.
        Reports the statistics of the selected metric.

        Args:
            test_prc: (float) percentage of data used for test partition as hold out
            metric: (str) name of the metric to report
            iterations: (int) number of iterations of model training
            mean_decimals: (int) number of decimals in approximation of the mean
            std_decimals: (int) number of decimals in approximation of the std

        Returns:
            (pd Dataframe) Report that contains the evaluation of a metric mean and std on train and test sets.
        """

        self.fit(test_prc, iterations)
        return self.compute(metric, mean_decimals, std_decimals)
