import numpy as np
from sklearn.model_selection import train_test_split

def get_metric(X, y, model, test_size, seed, evaluator, pred_type):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed, stratify = y)
    model = model.fit(X_train, y_train)
    if pred_type == 'proba':
        metric_test = evaluator(y_test, model.predict_proba(X_test)[:,1])
        metric_train = evaluator(y_train, model.predict_proba(X_train)[:,1])
    elif pred_type == 'class':
        metric_test = evaluator(y_test, model.predict(X_test))
        metric_train = evaluator(y_train, model.predict(X_train))
    else:
        raise ValueError(f'Unsupported prediction type was passed: {pred_type}')

    return [metric_train, metric_test]
