from probatus.models import lending_club_model
from probatus.datasets import lending_club
from sklearn.metrics import roc_auc_score

import numpy as np
from probatus.metric_uncertainty import VolatilityEstimation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from probatus.models import lending_club_model
from probatus.datasets import lending_club
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.stats
from scipy import stats


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:06:52 2018

@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison

updated: Raul Sanchez-Vazquez
"""



# loading a dummy model
model = lending_club_model()
data = lending_club(modelling_mode = False)[0]
y = data[['default']]
X = data.drop(['id', 'loan_issue_date','default'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)
model = model.fit(X_train, y_train)
alpha = .95
y_pred = model.predict_proba(X_test)[:,1]
y_true = y_test.values.flatten()

auc, auc_cov = delong_roc_variance(
    y_true,
    y_pred)

auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(
    lower_upper_q,
    loc=auc,
    scale=auc_std)

ci[ci > 1] = 1

print('AUC:', auc)
print('AUC COV:', auc_cov)
print('95% AUC CI:', ci)


def slicer(x, y, k):
    
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

def measurements(x, y, model):
    
    auc_test = []
    auc_delta = []

    for trian_x, trian_y in zip(x, y):
        if np.mean(trian_y) < 1 and np.mean(trian_y) > 0:
            model = model.fit(trian_x, trian_y.flatten())
            y_pred_test = model.predict_proba(X_test)[:,1]
            y_pred_train = model.predict_proba(trian_x)[:,1]
            y_true = y_test.values.flatten()
            auc_i_test = roc_auc_score(y_true, y_pred_test)
            auc_i_train = roc_auc_score(trian_y.flatten(), y_pred_train)

            auc_test.append(auc_i_test)
            auc_delta.append(auc_i_train - auc_i_test)
            
        else:
            print('split does not have all classes of y')
    
    return np.mean(auc_test), np.var(auc_test), np.mean(auc_delta), np.var(auc_delta)

credit_df, X_train, X_test, y_train, y_test = lending_club(modelling_mode = True)    

model = lending_club_model()

V_s = []
V_mean = []
delta_m = []
delta_s = []

max_k = int(np.floor(y_train.shape[0]/(y_train.shape[0] * y_train.values.mean())))

for k in range(2,max_k + 1):
    
    print(f'running executiong for {k} folds')

    x_slice, y_slice = slicer(X_train.values, y_train.values, k)
    approx_m, approx_v, delta_m_i, delta_v_i = measurements(x_slice, y_slice, model)
    V_s.append(approx_v)
    V_mean.append(approx_m)
    delta_m.append(delta_m_i)
    delta_s.append(delta_v_i)

    
print(f'the estimated population mean variance is {str(np.mean(V_mean))}')
print(f'the estimated population variance of variance is {str(np.mean(V_s))}')

print(f'the estimated mean delta between tain and test {str(np.mean(delta_m))}')
print(f'the estimated variance delta between tain and test {str(np.mean(delta_s))}')

delta_exp = np.random.normal(np.mean(delta_m), np.sqrt(np.mean(delta_s)), 1000000)
plt.hist(delta_exp)
plt.show()


# loading original data
data = lending_club(modelling_mode = False)[0]
y = data[['default']]
X = data.drop(['id', 'loan_issue_date','default'], axis = 1)

# defining the metrics we are interested in
evaluators =  {'AUC' : [roc_auc_score,'proba']}

# declaring the evaluation class
checker = VolatilityEstimation(model, X, y, evaluators)

# running 1000 random samples with 40% of data assigned to test partition
checker.estimate(0.4,1000)

auc_boot = np.random.normal(np.mean(checker.metrics_list['AUC'][:,1]), np.sqrt(np.var(checker.metrics_list['AUC'][:,1])), 10000) 
v_exp = np.random.normal(np.mean(V_mean), np.sqrt(np.mean(V_s)), 10000)
conf_ind = np.random.normal(auc, auc_std, 10000)

import matplotlib.pyplot as plt

plt.hist(v_exp)
plt.hist(conf_ind)
plt.hist(auc_boot)
plt.show()

