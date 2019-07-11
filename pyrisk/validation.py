import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def propensity_check(X1, X2, model):
    """Calculates and returns the AUC score.
    
    Args:
        X1 (numpy.ndarray)          : first sample
        X2 (numpy.ndarray)          : second sample
        model (sklearn.ensemble)    : model name 

    Returns: 
        AUC score                   : float  
    """
    X = np.concatenate([X1,X2])
    y = np.zeros(len(X1) + len(X2))
    y[len(X1):] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size=0.33, 
                                                        random_state=42)
    model.fit(X_train,y_train)
    score = model.predict_proba(X_test)[:,1]
    
    return roc_auc_score(y_test, score)
