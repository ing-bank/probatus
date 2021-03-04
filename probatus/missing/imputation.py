# Copyright (c) 2021 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from probatus.utils import preprocess_data, preprocess_labels,BaseFitComputePlotClass,get_single_scorer
from  sklearn.model_selection import cross_val_score
from  sklearn.pipeline import Pipeline
from  sklearn.impute import SimpleImputer
from  sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

class CompareImputationStrategies(BaseFitComputePlotClass):
    """
    Comparison of various imputation stragegies that can be used for imputation 
    of missing values. 
    The aim of this class is to present the model performance based on imputation
    strategies and choosen model.
    For models like XGBoost & LighGBM which have capabilities to handle misisng values by default
    the model performance with no imputation will be shown as well.
    The missing values categorical features are filled with `missing` and an missing indicator is
    added.

    Usage E.g.
    ```python

    #Import the class
    from probatus.missing.imputation import CompareImputationStrategies
    #Create the strategies.
    strategies = {
       'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
       'Simple Mean Imputer' : SimpleImputer(strategy='mean',add_indicator=True),
       'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
       sample_posterior=True),
       'KNN' : KNNImputer(n_neighbors=3)
    #Create a classifier.
    clf = lgb.LGBMClassifier()
    #Create the comparision of the imputation strategies.
    cmp = CompareImputationStrategies(
        clf=clf,
        strategies=strategies,
        cv=5,
        model_na_support=True)

    cmp.fit_compute(X_missing,y)
    #Plot the results.
    cmp.plot()

    <img src="../img/imputation_comparision.png" width="500" />
   }

    ```

    """
    def __init__(self,clf,strategies,scoring='roc_auc',cv=None,model_na_support=True,n_jobs=-1,verbose=0,
                random_state=None):
        """
        Initialise the class.

        Args :
            clf(model object):
                A binary classification model, that will used to evaluate various imputation strategies.

            strategies (dictionary of sklearn.impute objects):
                Dictionary containing the sklearn.impute objects.
                e.g.
                
                strategies = {'KNN' : KNNImputer(n_neighbors=3),
                'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
                'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
                sample_posterior=True)}
                This allows you to have fine grained control over the imputation method.

            scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers, optional):
                Metrics for which the score is calculated. It can be either a name or list of names metric names and
                needs to be aligned with predefined [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric.

            model_na_support(boolean): default True
                If the classifier supports missing values by default e.g. LightGBM,XGBoost etc. 
                If True an default comparison `Model Imputation` will be added indicating the results without any explict imputation. 
                If False only the provided strategies will be used.
            
            n_jobs (int, optional):
                Number of cores to run in parallel while fitting across folds. None means 1 unless in a
                `joblib.parallel_backend` context. -1 means using all processors.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - nether prints nor warnings are shown
                - 1 - 50 - only most important warnings regarding data properties are shown (excluding SHAP warnings)
                - 51 - 100 - shows most important warnings, prints of the feature removal process
                - above 100 - presents all prints and all warnings (including SHAP warnings).
            
            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """
        self.clf = clf
        self.model_na_support = model_na_support
        self.scorer = get_single_scorer(scoring)
        self.strategies = strategies
        if cv is None:
            self.cv = 5
        else :
            self.cv = cv
        self.verbose = verbose

        self.n_jobs = n_jobs
        
        if random_state is None:
            self.random_state = 42
        else:
            self.random_state = random_state

        self.fitted = False
        self.report = pd.DataFrame([])

    def __repr__(self):
        return "Imputation comparision for {}".format(self.clf.__class__.__name__)


    def fit(self, X, y,column_names=None,categorical_columns='auto'):
        """
        Calculates the cross validated results for various imputation strategies.
        
        Args:
            X (pd.DataFrame):
                input variables.

            y (pd.Series):
                target variable.

            column_names (None, or list of str, optional):
                List of feature names for the dataset. 
                If None, then column names from the X dataframe are used.
            
            categorical_features (None, or list of str, optional):default=auto
                List of categorical features.The imputation strategy for categorical 
                is different that compared to numerical features. If auto try to infer
                the categorical columns based on 'object' and 'category' datatypes.
        """
        #Place holder for results.
        results = []

        self.X, self.column_names = preprocess_data(X, column_names=column_names,
                                                          verbose=self.verbose)
        self.y = preprocess_labels(y, index=self.X.index, verbose=self.verbose)
                                                         

        #Identify categorical features if not explicitly specified.
        if 'auto' in categorical_columns:
            X_cat = X.select_dtypes(include=['category','object'])
            categorical_columns = X_cat.columns.to_list()
            for column in categorical_columns:
                X[column] = X[column].astype('category')
        else :
            #Check if the passed columns are in the dataframe.
            assert categorical_columns in X.columns,"All categorical columns not in the dataframe."
            X_cat = X[categorical_columns]
        #Identify the numeric columns.Numeric columns are all columns expect the categorical
        # columns
        X_num = X.drop(columns = categorical_columns,inplace=False)
        numeric_columns = X_num.columns.to_list()
        
        for strategy in self.strategies:

            numeric_transformer = Pipeline(steps=[
                    ('imputer', self.strategies[strategy])])

            categorical_transformer = Pipeline(steps=[
                    ('imp_cat',SimpleImputer(strategy='constant',fill_value='missing',add_indicator=True)),
                    ('ohe_cat',OneHotEncoder(handle_unknown='ignore')),
                ])

            preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_columns),
                        ('cat', categorical_transformer, categorical_columns)],
                        remainder='passthrough')

            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', self.clf)])
                
            imputation_results = cross_val_score(
                clf,
                X,
                y,
                scoring=self.scorer.scorer,
                cv=self.cv,
                n_jobs = self.n_jobs)

            temp_results = {
                    'strategy' : strategy,
                    'score': np.round(np.mean(imputation_results),3),
                    'std':np.round(np.std(imputation_results),3),
                }
            results.append(temp_results)
        #If model supports missing values by default, then calculate the scores 
        #on raw data without any imputation. 
        if self.model_na_support :
            categorical_transformer = Pipeline(steps=[
                    ('ohe_cat',OneHotEncoder(handle_unknown='ignore')),
                ])

            preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', categorical_transformer, categorical_columns)],
                        remainder='passthrough')

            self.clf = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', self.clf)])

            imputation_results = cross_val_score(
                self.clf,
                X,
                y,
                scoring=self.scorer.scorer,
                cv=self.cv,
                n_jobs = self.n_jobs
                )

            temp_results = {
                    'strategy' : 'Model Imputation',
                    'score': np.round(np.mean(imputation_results),3),
                    'std':np.round(np.std(imputation_results),3),
                }
            results.append(temp_results)


        self.report = pd.DataFrame(results)
        self.report.sort_values(by='score',inplace=True)
        self.fitted = True
        return self
        
        
    def compute(self,return_scores=True):
        """
        Compute method.
        """
        self._check_if_fitted()
        if return_scores :
            return self.report

    def fit_compute(self, X, y,column_names=None,categorical_columns='auto'):
        """
        Calculates the cross validated results for various imputation strategies.
        
        Args:
            X (pd.DataFrame):
                input variables.

            y (pd.Series):
                target variable.

            column_names (None, or list of str, optional):
                List of feature names for the dataset. 
                If None, then column names from the X dataframe are used.
            
            categorical_features (None, or list of str, optional):default=auto
                List of categorical features.The imputation strategy for categorical 
                is different that compared to numerical features. If auto try to infer
                the categorical columns based on 'object' and 'category' datatypes.
        """
        self.fit(X,y,
        column_names=column_names,
        categorical_columns=categorical_columns
        )
        return self.compute()


    def plot(self,show=True):
        """
        Plot the results for imputation.
        """
        imp_methods = list(self.report['strategy'])
        performance = list(self.report['score'])
        std_error = list(self.report['std'])

        y_pos = [i for i, _ in enumerate(imp_methods)]  
        x_spacing = 0.01
        y_spacing = 2*x_spacing
        plt.barh(
            y_pos, 
            performance,
            xerr=std_error,
            align='center',
            color=np.random.rand(len(performance),3))

        for index, value in enumerate(performance):
            plt.text(value+x_spacing ,index+y_spacing, str(value),rotation=45)
        plt.yticks(y_pos, imp_methods)
        plt.xlabel(f"Metric ({(self.scorer.metric_name).replace('_',' ').upper()})")
        plt.title("Imputation Techniques")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close()