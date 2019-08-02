import numpy as np
from sklearn.model_selection import train_test_split

def get_metric(X, y, model, test_size, seed, evaluator, pred_type):
    """
    Draws random train/test sample from the data using random seed and calculates metric of interest.

    Args:
        X: pdDataFrame or nparray with features
        y: pdDataFrame or nparray with targets
        test_size: float fraction of data used for testing the model
        seed: int randomized seed used for splitting data
        evaluator : function used for calculating evaluation metric
        pred_type : string form of prediction which is used for obtaining evaluation metric.

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
