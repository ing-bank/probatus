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

from probatus.utils import BaseFitComputeClass,BaseFitComputePlotClass
from  sklearn.model_selection import cross_val_score
from  sklearn.pipeline import make_pipeline,Pipeline
from  sklearn.impute import SimpleImputer
from  sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import numpy as np 
class CompareImputationStrategies(BaseFitComputeClass):
    """
    Comparison of various imputation stragegies
    that can be used for imputation of missing values. 

    Args :

    """
    def __init__(self,clf,strategies,scoring='roc_auc',cv=5,verbose=0):
        """
        Initialise the class 

        Args :
            clf(model object):
                Binary classification model.

            scoring (string, list of strings, probatus.utils.Scorer or list of probatus.utils.Scorers, optional):
                Metrics for which the score is calculated. It can be either a name or list of names metric names and
                needs to be aligned with predefined [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric.

            strategies (dictionary of sklearn.impute objects):
                Dictionary containing the sklearn.impute objects.
                #TODO Add more documentation.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - nether prints nor warnings are shown
                - 1 - 50 - only most important warnings regarding data properties are shown (excluding SHAP warnings)
                - 51 - 100 - shows most important warnings, prints of the feature removal process
                - above 100 - presents all prints and all warnings (including SHAP warnings).
        """
        self.clf = clf
        self.scoring = scoring
        self.strategies = strategies
        self.cv = cv
        self.verbose = verbose
        self.results = {}
        
    def fit(self, X, y,column_names=None,class_names=None,categorical_columns='auto'):
        """
        Calculates score
        
        Args:
            X (pd.DataFrame):
                input variables.

            y (pd.Series):
                target variable.

            column_names (None, or list of str, optional):
                List of feature names for the dataset. If None, then column names from the X_train dataframe are used.

            class_names (None, or list of str, optional):
                List of class names e.g. ['neg', 'pos']. If none, the default ['Negative Class', 'Positive Class'] are
                used.
            
            categorical_features ((None, or list of str, optional):deafault=auto
                List of categorical features to consider.
                The imputation strategy for categorical is different
                that compared to numerical features.
        """
        #Identify categorical features if not explicitly specified.
        if 'auto' in categorical_columns:
            X_cat = X.select_dtypes(include=['category','object'])
            categorical_columns = X_cat.columns.to_list()
            for column in categorical_columns:
                X[column] = X[column].astype('category')
        else :
            #Check if the passed columns are in the dataframe.
            X_cat = X[categorical_columns]
        
        X_num = X.drop(columns = categorical_columns,inplace=False)
        numeric_columns = X_num.columns.to_list()

        #Add the No imputation to strategy.
        self.strategies['No Imputation'] = None 

        for strategy in self.strategies:

            if 'No Imputation' in strategy:
                
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
                scoring=self.scoring,
                cv=self.cv)

                self.results[strategy] = imputation_results
                
            else :

                numeric_transformer = Pipeline(steps=[
                    ('imputer', self.strategies[strategy])])

                categorical_transformer = Pipeline(steps=[
                    ('imp_cat',SimpleImputer(strategy='most_frequent',add_indicator=True)),
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
                scoring=self.scoring,
                cv=self.cv)

                self.results[strategy] = imputation_results

        
        
    def compute(self):
        """
        Compute class

        """

    def fit_compute(self):
        """
        Fit & compute class
        """

    def show(self):
        """
        Show the results.
        """
        self._plot_results()

    def _get_no_imputer_scores(self,X,y):
        """
        Calculate the results without any imputation strategy.
        Args :
            X(pd.DataFrame) : Dataframe for X
            y(pd.Series) : Target 
        """
        no_imputer_scores = cross_val_score(
            self.clf,
            X,
            y,
            scoring=self.scoring,
            cv=self.cv)
    
        return no_imputer_scores

    def _plot_results(self):
        """
        Plot the results.
        """

        imp_methods = []
        performance = []
        std_error = []
        cmap=[]
    
        for k,v in self.results.items():
            imp_methods.append(k)
            performance.append(np.round(np.mean(v),4))
            std_error.append(np.round(np.std(v),4))
            cmap.append(np.random.rand(3,))
            

        y_pos = np.arange(len(imp_methods))    

        plt.barh(y_pos, performance, xerr=std_error,align='center',color=cmap)
        for index, value in enumerate(performance):
            plt.text(value, index, str(value))
        plt.yticks(y_pos, imp_methods)
        plt.xlabel('Metric')
        plt.title('Imputation Techniques')

        plt.show()
    

   
