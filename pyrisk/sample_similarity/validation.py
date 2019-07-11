import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

def propensity_check(X1, X2):
    """Calculates and returns the AUC score.
    
    Args:
        X1 (numpy.ndarray)          : first sample
        X2 (numpy.ndarray)          : second sample

    Returns: 
        AUC score                   : float  
    """   
    check_and_fill_missings(X1)
    check_and_fill_missings(X2)
    
    X = np.concatenate([X1,X2])
    y = np.zeros(len(X1) + len(X2))
    y[len(X1):] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size=0.33, 
                                                        random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    
    score = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, score)
    importances = get_feature_importance(model)
    
    return auc, importances


def get_feature_importance(model):
    """Returns feature importance for a given model

    Args:
        model (object): model object
    Returns:
        importances (pandas series): feature importance 
    """
    feat_importances = pd.Series(model.feature_importances_)
    
    return feat_importances

        
def check_and_fill_missings(X, impute_method=True):
    """
    Checks if for missing values: if there are and impute_method=True 
    it fills them with zero

    Args:
        X (numpy array):
        impute_method (bool)
    Returns:
        X (numpy array)
    """
    if np.isnan(X).any()==True:
        warnings.warn("You have missing values in your sample", 
                      DeprecationWarning, 
                      stacklevel=2)
    if impute_method:
        warnings.warn("Going to temporary solution: filling missing values with zero", 
                      DeprecationWarning, 
                      stacklevel=2)
        X[np.isnan(X)] = 0
    return X


            
