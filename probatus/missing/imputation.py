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
from  sklearn.pipeline import make_pipeline
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
        
    def fit(self, X, y,column_names=None,class_names=None):
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
        """
        #Add the No imputation to strategy.
        self.strategies['No Imputation'] = None 
        
        for strategy in self.strategies:

            if 'No Imputation' in strategy :
                imputation_results = self._get_no_imputer_scores(X,y)
                self.results[strategy] = imputation_results
            else :
                imputation_results = self.get_scores_for_imputer(
                    imputer = self.strategies[strategy],
                    X=X,
                    y=y)
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

        for k,v in self.results.items():
            print(f'{k}: {np.mean(v)} +/- {np.std(v)}')

    def _get_no_imputer_scores(self,X,y):
        """
        Calculate the results without any imputation.
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

    def get_scores_for_imputer(self,imputer,X,y):
        """
        Calculate the results with an imputer.
        args :
            imputer(sklearn.imputer) : The imputer object to use for imputation.
            X(pd.DataFrame) : Dataframe for X
            y(pd.Series) : Target 
        returns :
            impute_scores : 

        """
        
        estimator = make_pipeline(imputer,self.clf)

        impute_scores = cross_val_score(
            estimator,
            X,
            y,
            scoring=self.scoring,
            cv=self.cv)

        return impute_scores
