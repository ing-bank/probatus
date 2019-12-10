import pytest
import numpy as np
from probatus.metric_uncertainty import VolatilityEstimation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from probatus.datasets import lending_club
from sklearn.ensemble import RandomForestClassifier

#### Fixtures 

@pytest.fixture
def model():
    return RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=6, max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
        oob_score=False, random_state=0, verbose=0, warm_start=False)

@pytest.fixture
def club_data():
    # loading original data
    data = lending_club(modelling_mode = False)[0]
    return data

@pytest.fixture
def X(club_data):
    return club_data.drop(['id', 'loan_issue_date','default'], axis = 1)

@pytest.fixture
def y(club_data):
    return club_data[['default']]
    
@pytest.fixture
def bootseed_checker(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_seed')
    checker.estimate(0.4,10) 
    return checker

#### Tests     

def test_metric_uncertainty_reproducibility(X, y, model, bootseed_checker):
    """
    Running the VolatilityEstimation twice on same data and model
    should produce the same results.
    """
    
    # boot_seed 
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker2 = VolatilityEstimation(model, X, y, evaluators, 'boot_seed')
    checker2.estimate(0.4,10)
    
    auc_is_equal = bootseed_checker.metrics_list['AUC'] == checker2.metrics_list['AUC']
    assert auc_is_equal.all(), "boot_seed not deterministic"
    
    # boot_global
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4, 10)
    checker2 = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker2.estimate(0.4, 10) 
    
    auc_is_equal = checker.metrics_list['AUC'] == checker2.metrics_list['AUC']
    assert auc_is_equal.all(), "boot_global not deterministic"
    
    # delong
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4, 10)
    checker2 = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker2.estimate(0.4, 10) 

    auc_is_equal = checker.metrics_list['AUC'] == checker2.metrics_list['AUC']
    assert auc_is_equal.all(), "delong not deterministic"
    

def test_metric_uncertainty_metrics_length_seed(X, y, model, bootseed_checker):
    """
    Make sure all metrics are properly returned
    """
    assert list(bootseed_checker.metrics_list.keys()) == ['AUC', 'ACC']

def test_metric_uncertainty_metrics_length_global(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4)

    assert list(checker.metrics_list.keys()) == ['AUC']

def test_metric_uncertainty_metrics_length_delong(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4)

    assert list(checker.metrics_list.keys()) == ['AUC']

def test_metric_uncertainty_array_length_seed(X, y, model, bootseed_checker):
    assert bootseed_checker.metrics_list['ACC'].shape[0] == 10
    assert bootseed_checker.metrics_list['AUC'].shape[0] == 10


def test_metric_uncertainty_array_length_global(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4)

    assert checker.metrics_list['AUC'].shape[0] > 5
    assert checker.metrics_list['ACC'].shape[0] > 5

def test_metric_uncertainty_array_length_delong(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4)

    assert checker.metrics_list['AUC'].shape[1] == 6
    assert checker.metrics_list['AUC'].shape[0] == 1

def test_metric_uncertainty_metric_values_seed(X, y, model, bootseed_checker):
    assert np.mean(bootseed_checker.metrics_list['ACC']) > 0
    assert np.mean(bootseed_checker.metrics_list['AUC']) > 0

def test_metric_uncertainty_metric_values_global(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'boot_global')
    checker.estimate(0.4)

    assert np.mean(checker.metrics_list['ACC']) > 0
    assert np.mean(checker.metrics_list['AUC']) > 0

def test_metric_uncertainty_metric_values_delong(X, y, model):
    evaluators =  {'AUC' : [roc_auc_score,'proba'], 'ACC' : [accuracy_score,'class']}
    checker = VolatilityEstimation(model, X, y, evaluators, 'delong')
    checker.estimate(0.4)

    assert np.mean(checker.metrics_list['AUC']) > 0
