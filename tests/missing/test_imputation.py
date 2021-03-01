#Code to test the imputation strategies.
from probatus.missing.imputation import CompareImputationStrategies
from tests.utils.test_missing import generate_MCAR,generate_MNAR
import pandas as pd 
from sklearn.datasets import make_classification
import lightgbm as lgb 
import xgboost as xgb 
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import KNNImputer,SimpleImputer,IterativeImputer
from feature_engine.imputation import RandomSampleImputer

def test_imputation():
   X,y = make_classification(
       n_samples=1000, 
       n_features=20,
       class_sep = 0.3)
   X_missing = generate_MCAR(pd.DataFrame(X),missing=0.3)
   #Initialize the classifier
   clf = lgb.LGBMClassifier()
   #Create strategies for imputation.
   strategies = {
       'KNN' : KNNImputer(n_neighbors=3),
       'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
       'Simple Mean Imputer' : SimpleImputer(strategy='mean',add_indicator=True),
       'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
       sample_posterior=True),
       'Random Imputer': RandomSampleImputer()
       
   }
   cmp = CompareImputationStrategies(clf=clf,strategies=strategies,cv=10)
   cmp.fit(X_missing,y)
   cmp.show()

  
   

   


if __name__ == '__main__':
    test_imputation()