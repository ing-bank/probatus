import numpy as np
from pyrisk.metric_uncertainty import VolatilityEstimation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from pyrisk.models import lending_club_model
from pyrisk.datasets import lending_club

def test_metric_uncertainty_metrics_length():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators)
    checker.estimate(0.4,10)

    assert list(checker.metrics_list.keys()) == ['AUC', 'ACC']

def test_metric_uncertainty_array_length():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators)
    checker.estimate(0.4,10)

    assert checker.metrics_list['ACC'].shape[0] == 10
    assert checker.metrics_list['AUC'].shape[0] == 10

def test_metric_uncertainty_metric_values():

    # loading a dummy model
    model = lending_club_model()

    # loading original data
    data = lending_club(modelling_mode = False)[0]
    y = data[['default']]
    X = data.drop(['id', 'loan_issue_date','default'], axis = 1)
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators)
    checker.estimate(0.4,10)

    assert np.mean(checker.metrics_list['ACC']) > 0
    assert np.mean(checker.metrics_list['AUC']) > 0