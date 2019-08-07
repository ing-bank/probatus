import numpy as np
import matplotlib.pyplot as plt
from pyrisk.metric_uncertainty.metric import get_metric
from pyrisk.metric_uncertainty.sampling import stratified_random
from joblib import Parallel, delayed
from pyrisk.utils import assure_numpy_array


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

    """
    def __init__(self, model, X, y, evaluator, n_jobs=1):
        self.model = model
        self.X = assure_numpy_array(X)
        self.y = assure_numpy_array(y)
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.metrics_list = {}

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
            random_seeds = np.random.random_integers(0, 999999, iterations)
            results = Parallel(n_jobs=self.n_jobs)(delayed(get_metric)(self.X, self.y, self.model, test_prc, i, self.evaluator[evaluator_i][0], self.evaluator[evaluator_i][1]) for i in random_seeds)
            self.metrics_list[evaluator_i] = np.array(results)

    def reporting(self, metric):
        """
        Prints summary statistics of the metric 

        Args:
            metric: str name of the metric to report

        """        
        metric_data = self.metrics_list[metric]
        print(f'Mean of metric on train is {round(np.mean(metric_data[:,0]),2)}')
        print(f'Mean of metric on test is {round(np.mean(metric_data[:,1]),2)}')
        print(f'Standard Deviation of metric on train is {round(np.std(metric_data[:,0]),2)}')
        print(f'Standard Deviation of metric on test is {round(np.std(metric_data[:,1]),2)}')

    def plot(self, metric):
        """
        Plots detribution of the metric 

        Args:
            metric: str name of the metric to report

        """  
        metric_data = self.metrics_list[metric]
        plt.hist(metric_data[:,0], alpha=0.5, label='Train')
        plt.hist(metric_data[:,1], alpha=0.5, label='Test')
        plt.legend(loc='upper right')
        plt.show()
