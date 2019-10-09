import numpy as np
from sklearn.model_selection import StratifiedKFold

def max_folds(y):
    """
    Calculate maximum number of stratified folds whih can be created from the data

    Args:
        y: nparray with targets

    Returns: 
        k: int max number of stratitfied folds

    """
    k = int(np.floor(y.shape[0]/(y.shape[0] * np.mean(y))))

    return k

def slicer(x, y, k):
    """
    Slice data into k independent folds

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        k : int number of folds
    Returns: 
        x_slices: nparray sliced independent features folds
        y_slices: nparray sliced independent targets folds

    """    
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    
    folds_train = []
    folds_test = []

    for train_index, test_index in skf.split(x, y):
        folds_train.append(train_index)
        folds_test.append(test_index) 
        
    indexes = np.append(folds_train[0],folds_test[0])
    x = x[indexes]
    y = y[indexes]
    
    x_slices = np.array_split(x,k)
    y_slices = np.array_split(y,k)
    return x_slices, y_slices