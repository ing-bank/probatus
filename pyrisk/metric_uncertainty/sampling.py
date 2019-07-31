import numpy as np

def stratified_random(x, y, size):
    """
    Draws N random samples from the data to create new train/test splits and calculate metric of interest.
    After collecting multiple metrics per split, summary statistics of the metic are reported.
    This provides uncertainty levels around the metric if the train/test data is resampled.  

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        evaluator : dict with name of the metric as a key and array holding [scoring function, scoring type]
            e.g. {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}

    Returns: 
        statistic value and p_value (if available, e.g. not for PSI)

    """
    idnex_1 = np.argwhere(y == 1).flatten()
    index_0 = np.argwhere(y == 0).flatten()

    label_1 = x[idnex_1]
    label_0 = x[index_0]

    fraction = np.mean(y)

    sample_1 = label_1[np.random.randint(low = 0, high = label_1.shape[0], size = int(size*fraction))]
    sample_0 = label_0[np.random.randint(low = 0, high = label_0.shape[0], size = int(size*(1-fraction)))]

    y_sample = np.concatenate((np.repeat(1,sample_1.shape[0]),np.repeat(0,sample_0.shape[0])))
    x_sample = np.concatenate((sample_1,sample_0))

    return x_sample, y_sample