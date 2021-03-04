#Code to test the imputation strategies.
from probatus.missing.imputation import CompareImputationStrategies
import lightgbm as lgb 
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import KNNImputer,SimpleImputer,IterativeImputer
from feature_engine.imputation import RandomSampleImputer
import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope='function')
def X():
    return pd.DataFrame({'col_1': [1, np.nan, 1, 1, np.nan, 1, 1, 0],
                         'col_2': [0, 0, 0, np.nan, 0, 0, 0, 1],
                         'col_3': [1, 0, np.nan, 0, 1, np.nan, 1, 0], 
                         'col_4': ['A', 'B', 'A', np.nan, 'B', np.nan, 'A', 'A']}, index=[1, 2, 3, 4, 5, 6, 7, 8])

@pytest.fixture(scope='function')
def y():
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=[1, 2, 3, 4, 5, 6, 7, 8])

def test_imputation_boosting(X,y,capsys):

   #Create strategies for imputation.
   strategies = {
       'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
       'Simple Mean Imputer' : SimpleImputer(strategy='mean',add_indicator=True),
       'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
       sample_posterior=True),
       'KNN' : KNNImputer(n_neighbors=3),
   }
   #Initialize the classifier
   clf = lgb.LGBMClassifier()
   cmp = CompareImputationStrategies(clf=clf,strategies=strategies,cv=3,model_na_support=True)
   report = cmp.fit_compute(X,y)
   cmp.plot(show=False)
   
   assert cmp.fitted == True
   cmp._check_if_fitted()
   assert report.shape[0]==5

   # Check if there is any prints
   out, _ = capsys.readouterr()
   assert len(out) == 0

def test_imputation_linear(X,y,capsys):
    
   #Create strategies for imputation.
   strategies = {
       'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
       'Simple Mean Imputer' : SimpleImputer(strategy='mean',add_indicator=True),
       'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
       sample_posterior=True),
       'KNN' : KNNImputer(n_neighbors=3),
   }
   #Initialize the classifier
   clf = LogisticRegression()
   cmp = CompareImputationStrategies(clf=clf,strategies=strategies,cv=3,model_na_support=False)
   report = cmp.fit_compute(X,y)
   cmp.plot(show=False)
   
   assert cmp.fitted == True
   cmp._check_if_fitted()
   assert report.shape[0]==4

   # Check if there is any prints
   out, _ = capsys.readouterr()
   assert len(out) == 0