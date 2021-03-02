#Code to test the imputation strategies.
from probatus.missing.imputation import CompareImputationStrategies
from tests.utils.test_missing import generate_MCAR,generate_MNAR
import pandas as pd 
from sklearn.datasets import make_classification
import lightgbm as lgb 
import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import KNNImputer,SimpleImputer,IterativeImputer
from feature_engine.imputation import RandomSampleImputer
from sklearn.preprocessing import KBinsDiscretizer
import string
import fire

def test_imputation(choice=3):
   X,y = get_data(n_samples=1000,n_numerical=10,n_category=5)
   X_missing = generate_MCAR(X,missing=0.2)

   strategies = {
       'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
       'Simple Mean Imputer' : SimpleImputer(strategy='mean',add_indicator=True),
       'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
       sample_posterior=True),
       'KNN' : KNNImputer(n_neighbors=3),
       'Random Imputer': RandomSampleImputer()

   }

   #Initialize the classifier
   print(f'Using choice {choice}')
   if choice == 1:
       clf = RandomForestClassifier()
       cmp = CompareImputationStrategies(
           clf=clf,
           strategies=strategies,
           cv=5,
           model_na_support=False)
   if choice == 2  :
       clf = xgb.XGBClassifier()
       cmp = CompareImputationStrategies(
           clf=clf,
           strategies=strategies,
           cv=5,
           model_na_support=True)
   if choice == 3 :
       clf = lgb.LGBMClassifier()
       cmp = CompareImputationStrategies(
           clf=clf,
           strategies=strategies,
           cv=5,
           model_na_support=True)
   if choice == 4 :
       clf = LogisticRegression()
       cmp = CompareImputationStrategies(
           clf=clf,
           strategies=strategies,
           cv=5,
           model_na_support=False)

   #Create strategies for imputation.
   
  
   cmp.fit_compute(X_missing,y)
   cmp.plot()
   


def get_data(n_samples,n_numerical,n_category):
        """
        Returns a dataframe with numerical and categorical features.
        """
        no_vars = n_numerical + n_category
       
        X,y = make_classification(
            n_samples=n_samples, 
            n_features=no_vars, 
            random_state=123,class_sep=0.3)

        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile", )
        X[:,n_numerical:] = binner.fit_transform(X[:,n_numerical:])

        #Add column names.
        X = pd.DataFrame(X, columns=["f_"+str(i) for i in range(0,no_vars)])

        # Efficiently map values to another value with .map(dict)
        X.iloc[:,n_numerical:] = X.iloc[:,n_numerical:].apply(
            lambda x: x.map({i:letter for i,letter in enumerate(string.ascii_uppercase)}))
        
        return X,y

if __name__ == '__main__':
    fire.Fire(test_imputation)