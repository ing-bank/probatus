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
from sklearn.preprocessing import KBinsDiscretizer
import string

def test_imputation():
   X,y = get_data(n_samples=1000,n_numerical=10,n_category=5)
   X_missing = generate_MCAR(X,missing=0.2)
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


def get_data(n_samples,n_numerical,n_category):
        """
        Returns a dataframe with numerical and categorical features
        """
        no_vars = n_numerical + n_category
        # Create single dataset to avoid random effects
        # Only works for all informative features
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
    test_imputation()