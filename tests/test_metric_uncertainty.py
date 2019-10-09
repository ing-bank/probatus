import numpy as np
from probatus.metric_uncertainty import VolatilityEstimation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from probatus.models import lending_club_model
from probatus.datasets import lending_club

def test_metric_uncertainty_metrics_length_seed():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_seed')
    checker.estimate(0.4,10)

    assert list(checker.metrics_list.keys()) == ['AUC', 'ACC']

def test_metric_uncertainty_metrics_length_global():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4)

    assert list(checker.metrics_list.keys()) == ['AUC']

def test_metric_uncertainty_metrics_length_delong():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4)

    assert list(checker.metrics_list.keys()) == ['AUC']

def test_metric_uncertainty_array_length_seed():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_seed')
    checker.estimate(0.4,10)

    assert checker.metrics_list['ACC'].shape[0] == 10
    assert checker.metrics_list['AUC'].shape[0] == 10


def test_metric_uncertainty_array_length_global():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4)

    assert checker.metrics_list['AUC'].shape[0] > 5
    assert checker.metrics_list['ACC'].shape[0] > 5

def test_metric_uncertainty_array_length_delong():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4)

    assert checker.metrics_list['AUC'].shape[1] == 6
    assert checker.metrics_list['AUC'].shape[0] == 1

def test_metric_uncertainty_metric_values_seed():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_seed')
    checker.estimate(0.4,10)

    assert np.mean(checker.metrics_list['ACC']) > 0
    assert np.mean(checker.metrics_list['AUC']) > 0

def test_metric_uncertainty_metric_values_global():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4)

    assert np.mean(checker.metrics_list['ACC']) > 0
    assert np.mean(checker.metrics_list['AUC']) > 0

def test_metric_uncertainty_metric_values_delong():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4)

    assert np.mean(checker.metrics_list['AUC']) > 0