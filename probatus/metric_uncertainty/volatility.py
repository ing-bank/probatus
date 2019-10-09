import numpy as np
import matplotlib.pyplot as plt
from probatus.metric_uncertainty.metric import get_metric
from joblib import Parallel, delayed
from probatus.utils import assure_numpy_array
from probatus.metric_uncertainty import delong
from probatus.metric_uncertainty.utils import max_folds
from probatus.metric_uncertainty.sampling import stratified_random
from probatus.metric_uncertainty.utils import slicer
from probatus.metric_uncertainty.metric import get_metric_folds

class VolatilityEstimation(object):
    """
    Draws N random samples from the data to create new train/test splits and calculate metric of interest.
    After collecting multiple metrics per split, summary statistics of the metic are reported.
    This provides uncertainty levels around the metric if the train/test data is resampled.  

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        evaluator : dict with name of the metric as a key and array holding [scoring function, scoring type]
            e.g. {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
        n_jobs: number of cores to be used. -1 sets it to use all the free cores
        method: str type of estimation which should be used, currently supports:
                boot_global - bootstrap replicates with global estimation of the AUC uncertainty with non onverlapping resampling
                boot_seed - boostrap replicates with local estimation of the AUC uncertainty (fixed split) and overlapping resampling
                delong - delong estimator of the AUC uncertainty

    """
    def __init__(self, model, X, y, evaluator, method, n_jobs=1):
        self.model = model
        self.X = assure_numpy_array(X)
        self.y = assure_numpy_array(y)
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.metrics_list = {}
        self.method = method

    def estimate(self, test_prc, iterations = 1000):
        """
        Runs a parallel loop to sample multiple metrics using different train/test splits  

        Args:
            test_prc: flot percentage of data used for test partition
            iterations: int number of samples

        Returns: 
            Popultes dictionary with metrics data from sampling

        """
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
                                                                           self.evaluator[evaluator_i][1]) for i in random_seeds)
                self.metrics_list[evaluator_i] = np.array(results)
            
            elif self.method == 'boot_global':

                results = []
                X_train, X_test, y_train, y_test = stratified_random(self.X, self.y, test_prc)
                top_k = max_folds(y_train) 
                if top_k > 11:
                    top_k = 11
                for k in range(1,top_k + 1):

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
                self.metrics_list[evaluator_i] = np.array(results) 

            elif self.method == 'delong':    
                X_train, X_test, y_train, y_test = stratified_random(self.X, self.y, test_prc)
                model = self.model.fit(X_train, y_train)

                y_pred_test = model.predict_proba(X_test)[:,1]
                y_pred_train = model.predict_proba(X_train)[:,1]

                auc_train, auc_cov_train = delong.delong_roc_variance(y_train, y_pred_train)
                auc_test, auc_cov_test = delong.delong_roc_variance(y_test, y_pred_test)

                self.metrics_list[evaluator_i] = np.array([auc_train, auc_test,auc_train - auc_test, auc_cov_train, auc_cov_test, 0]).reshape(1,6)

    def reporting(self, metric):
        """
        Prints summary statistics of the metric 

        Args:
            metric: str name of the metric to report

        """        
        metric_data = self.metrics_list[metric]
        print(f'Mean of metric on train is {round(np.mean(metric_data[:,0]),2)}')
        print(f'Mean of metric on test is {round(np.mean(metric_data[:,1]),2)}')
        print(f'Mean of delta is {round(np.mean(metric_data[:,2]),2)}')

        if self.method == 'boot_seed':
            print(f'Standard Deviation of metric on train is {round(np.std(metric_data[:,0]),5)}')
            print(f'Standard Deviation of metric on test is {round(np.std(metric_data[:,1]),5)}')
            print(f'Standard Deviation of delta is {round(np.std(metric_data[:,2]),5)}')
        elif self.method == 'boot_global' or self.method == 'delong':
            print(f'Standard Deviation of metric on train is {round(np.mean(metric_data[:,3]),5)}')
            print(f'Standard Deviation of metric on test is {round(np.mean(metric_data[:,4]),5)}')           
            print(f'Standard Deviation of delta is {round(np.mean(metric_data[:,5]),5)}')

    def plot(self, metric):
        """
        Plots detribution of the metric 

        Args:
            metric: str name of the metric to report

        """  
        metric_data = self.metrics_list[metric]

        if self.method == 'boot_seed':
            train = metric_data[:,0]
            test = metric_data[:,1]
            delta = metric_data[:,2]
        elif self.method == 'boot_global' or self.method == 'delong':
            train = np.random.normal(np.mean(metric_data[:,0]), np.sqrt(np.mean(metric_data[:,3])), 10000)
            test = np.random.normal(np.mean(metric_data[:,1]), np.sqrt(np.mean(metric_data[:,4])), 10000)
            delta = np.random.normal(np.mean(metric_data[:,2]), np.sqrt(np.mean(metric_data[:,5])), 10000)

        plt.hist(train, alpha=0.5, label='Train')
        plt.hist(test, alpha=0.5, label='Test')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(delta, alpha=0.5, label='Delta')
        plt.legend(loc='upper right')
        plt.show()        