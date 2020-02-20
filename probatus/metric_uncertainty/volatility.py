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

    def estimate(self, test_prc, iterations=1000):
        """
        Runs a parallel loop to sample multiple metrics using different train/test splits  

        Args:
            test_prc: flot percentage of data used for test partition
            iterations: int number of samples

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
            self.results_df.loc[evaluator_i, 'mean_train'] = np.mean(metric_data[:,0])
            self.results_df.loc[evaluator_i, 'mean_test'] = np.mean(metric_data[:,1])
            self.results_df.loc[evaluator_i, 'mean_delta'] = np.mean(metric_data[:,2])

            if self.method == 'boot_seed':
                self.results_df.loc[evaluator_i, 'std_train'] = np.std(metric_data[:,0])
                self.results_df.loc[evaluator_i, 'std_test'] = np.std(metric_data[:,1])
                self.results_df.loc[evaluator_i, 'std_delta'] = np.std(metric_data[:,2])
            elif self.method == 'boot_global' or self.method == 'delong':
                self.results_df.loc[evaluator_i, 'std_train'] = np.std(metric_data[:,3])
                self.results_df.loc[evaluator_i, 'std_test'] = np.std(metric_data[:,4])
                self.results_df.loc[evaluator_i, 'std_delta'] = np.std(metric_data[:,5])


    def get_report(self, metric):
        """
        Reports the statistics of the selected metric 

        Args:
            metric: str name of the metric to report

        Returns:
            pd Dataframe that contains the statistics

        """

        results = self.results_df.loc[[metric]]

        for col in results.columns:
            if 'mean' in col:
                results[col] = results[col].round(2)
            else:
                results[col] = results[col].round(5)

        return results

    def plot(self, metric):
        """
        Plots detribution of the metric 

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
            train = np.random.normal(results['mean_train'], np.sqrt(results['std_train']), 10000)
            test = np.random.normal(results['mean_test'], np.sqrt(results['std_test']), 10000)
            delta = np.random.normal(results['mean_delta'], np.sqrt(results['std_delta']), 10000)

        plt.hist(train, alpha=0.5, label='Train')
        plt.hist(test, alpha=0.5, label='Test')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(delta, alpha=0.5, label='Delta')
        plt.legend(loc='upper right')
        plt.show()
