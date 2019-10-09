import numpy as np
from sklearn.model_selection import train_test_split

def stratified_random(x, y, size):
    """
    Draws N random samples from the data to create new train/test splits and calculate metric of interest.
    After collecting multiple metrics per split, summary statistics of the metic are reported.
    This provides uncertainty levels around the metric if the train/test data is resampled.  

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        size : int sample size

    Returns: 
        train and test splits

    """
    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = size)

    return X_train, X_test, y_train, y_test