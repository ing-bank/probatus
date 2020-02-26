import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from probatus.utils import assure_numpy_array

def get_metric(X, y, model, test_size, seed, scorers):
    """
    Draws random train/test sample from the data using random seed and calculates metric of interest.

    Args:
        X: (nparray) with features
        y: (nparray) with targets
        test_size: (float) fraction of data used for testing the model
        seed: (int) randomized seed used for splitting data
        scorers: (list of Scorers) list of Scorer objects used to score the trained model

    Returns: 
        (pd.Dataframe) Dataframe with results for a given model trained. Rows indicate the metric measured and columns
         ther esults

    """

    X = assure_numpy_array(X)
    y = assure_numpy_array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    model = model.fit(X_train, y_train)

    results_columns = ['metric_name', 'train_score', 'test_score', 'delta_score']
    results = pd.DataFrame([], columns=results_columns)

    for scorer in scorers:
        score_train = scorer.score(model, X_train, y_train)
        score_test = scorer.score(model, X_test, y_test)
        score_delta = score_train - score_test

        results = results.append(pd.DataFrame([[scorer.metric_name, score_train, score_test, score_delta]],
                                                    columns=results_columns))
    return results


def get_metric_folds(x, y, model, x_test, y_test, evaluator, pred_type):
    """
    Draws random train/test independant samples from the data using random seed and calculates metric of interest.

    Args:
        X: nparray with features
        y: nparray with targets
        X_test: nparray test set with features
        y_test: nparray test set with targets
        test_size: float fraction of data used for testing the model
        evaluator : function used for calculating evaluation metric
        pred_type : string form of prediction which is used for obtaining evaluation metric.

    Returns:
        list with metrics from tain, test and delta between the two

    """
    # TODO change this method to get multiple metrics simultaniously

    metrics_train = []
    metrics_test = []
    metrics_delta = []

    x = assure_numpy_array(x)
    y = assure_numpy_array(y)
    x_test = assure_numpy_array(x_test)
    y_test = assure_numpy_array(y_test)

    for train_x, train_y in zip(x, y):
        if np.mean(train_y) < 1 and np.mean(train_y) > 0:
            model = model.fit(train_x, train_y.flatten())

            if pred_type == 'proba':
                metric_test = evaluator(y_test, model.predict_proba(x_test)[:,1])
                metric_train = evaluator(train_y, model.predict_proba(train_x)[:,1])
                metric_delta = metric_train - metric_test
            elif pred_type == 'class':
                metric_test = evaluator(y_test, model.predict(x_test))
                metric_train = evaluator(train_y, model.predict(train_x))
                metric_delta = metric_train - metric_test
            else:
                raise ValueError(f'Unsupported prediction type was passed: {pred_type}')

            metrics_train.append(metric_train)
            metrics_test.append(metric_test)
            metrics_delta.append(metric_delta)

    return [np.mean(metrics_train),np.mean(metrics_test), np.mean(metrics_delta), np.var(metrics_train), np.var(metrics_test), np.var(metrics_delta)]
