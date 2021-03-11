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

from probatus.utils import (
    preprocess_data,
    preprocess_labels,
    BaseFitComputePlotClass,
    get_single_scorer,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ImputationSelector(BaseFitComputePlotClass):
    """
    Comparison of various imputation strategies that can be used for imputing
    missing values.
    The aim of this class is to present the model performance based on imputation
    strategies and a choosen model.
    For models like XGBoost & LighGBM which have capabilities to handle missing values by default
    the model performance with no imputation will be shown as well.
    The missing values categorical features are imputed with the value `missing` and an missing indicator is
    added.

    Example usage.
    ```python

    #Import the class

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from probatus.missing_values.imputation import ImputationSelector
    from probatus.utils.missing_helpers import generate_MCAR
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import KNNImputer,SimpleImputer,IterativeImputer
    from sklearn.datasets import make_classification

    #Create data with missing values.
    n_features = 10
    X,y = make_classification(n_samples=1000,n_features=n_features,random_state=123,class_sep=0.3)
    X = pd.DataFrame(X, columns=["f_"+str(i) for i in range(0,n_features)])
    X_missing = generate_MCAR(X,missing=0.2)

    #Create the strategies.
    strategies = {
       'Simple Median Imputer' : SimpleImputer(strategy='median',add_indicator=True),
       'Simple Mean Imputer' : SimpleImputer(strategy='mean',add_indicator=True),
       'Iterative Imputer'  : IterativeImputer(add_indicator=True,n_nearest_features=5,
       sample_posterior=True),
       'KNN' : KNNImputer(n_neighbors=3)}
    #Create a classifier.
    clf = lgb.LGBMClassifier()
    #Create the comparision of the imputation strategies.
    cmp = ImputationSelector(
        clf=clf,
        strategies=strategies,
        cv=5,
        model_na_support=True)

    cmp.fit_compute(X_missing,y)
    #Plot the results.
    cmp.plot()

    ```
    <img src="../img/imputation_comparision.png" width="500" />

    """

    def __init__(
        self,
        clf,
        strategies,
        scoring="roc_auc",
        cv=5,
        model_na_support=False,
        n_jobs=-1,
        verbose=0,
        random_state=None,
    ):
        """
        Initialise the class.

        Args :
            clf (binary classifier,sklearn.Pipeline):
                A binary classification model, that will used to evaluate various imputation strategies.

            strategies (dictionary of sklearn.impute objects or any other scikit learn compatible imputer.):
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

            model_na_support(boolean): default False
                If the classifier supports missing values by default e.g. LightGBM,XGBoost etc.
                If True an default comparison `No Imputation`  result will be added indicating the model performance without any explict imputation.
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
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.fitted = False
        self.report_df = pd.DataFrame(
            columns=[
                "strategy",
                f"{self.scorer.metric_name} score",
                f"{self.scorer.metric_name} std",
            ]
        )

    def __repr__(self):
        return "Imputation comparision for {}".format(self.clf.__class__.__name__)

    def fit(self, X, y, column_names=None):
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
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Place holder for results.
        results = []

        self.X, self.column_names = preprocess_data(
            X, column_names=column_names, verbose=self.verbose
        )
        self.y = preprocess_labels(y, index=self.X.index, verbose=self.verbose)

        # Identify categorical features.
        X_cat = X.select_dtypes(include=["category", "object"])
        categorical_columns = X_cat.columns.to_list()

        # Identify the numeric columns.Numeric columns are all columns expect the categorical columns
        
        X_num = X.drop(columns=categorical_columns, inplace=False)
        numeric_columns = X_num.columns.to_list()

        for strategy in self.strategies:

            numeric_transformer = Pipeline(
                steps=[("imputer", self.strategies[strategy])]
            )

            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imp_cat",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="missing",
                            add_indicator=True,
                        ),
                    ),
                    ("ohe_cat", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_columns),
                    ("cat", categorical_transformer, categorical_columns),
                ],
                remainder="passthrough",
            )

            model_pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", self.clf)]
            )

            temp_results = self._calculate_results(
                X, y, clf=model_pipeline, strategy=strategy
            )

            results.append(temp_results)

        # If model supports missing values by default, then calculate the scores
        # on raw data without any imputation.
        if self.model_na_support:

            categorical_transformer = Pipeline(
                steps=[
                    ("ohe_cat", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[("cat", categorical_transformer, categorical_columns)],
                remainder="passthrough",
            )

            model_pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", self.clf)]
            )

            temp_results = self._calculate_results(
                X, y, clf=model_pipeline, strategy="No Imputation"
            )
            results.append(temp_results)

        self.report_df = pd.DataFrame(results)
        self.report_df.sort_values(by=f"{self.scorer.metric_name} score", inplace=True)
        self.fitted = True
        return self

    def _calculate_results(self, X, y, clf, strategy):
        """
        Method to calculate the results for a particular imputation strategy.

        Args:
            X (pd.DataFrame):
                input variables.

            y (pd.Series):
                target variable.

            clf (binary classifier,sklearn.Pipeline):
                A binary classification model, that will used to evaluate various imputation strategies.

            strategy(string):
                Name of the strategy used for imputation.

        Returns:

            temp_df(dict) :  Dictionary containing the results of the evaluation.
        """
        imputation_results = cross_val_score(
            clf, X, y, scoring=self.scorer.scorer, cv=self.cv, n_jobs=self.n_jobs
        )

        temp_results = {
            "strategy": strategy,
            f"{self.scorer.metric_name} score": np.round(
                np.mean(imputation_results), 3
            ),
            f"{self.scorer.metric_name} std": np.round(np.std(imputation_results), 3),
        }

        return temp_results

    def compute(self):
        """
        Checks if fit() method has been run and computes the DataFrame with results of imputation for each
        strategy.

        Returns:
            (pd.DataFrame):
                 DataFrame with results of imputation for each strategy.
        """
        self._check_if_fitted()
        return self.report_df

    def fit_compute(self, X, y, column_names=None):
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

        Returns:
            (pd.DataFrame):
                DataFrame with results of imputation for each strategy.

        """
        self.fit(X, y, column_names=column_names)
        return self.compute()

    def plot(self, show=True, **figure_kwargs):
        """
        Generates plot of the performance of various imputation strategies.

        Args:
            show (bool, optional):
                If True, the plots are showed to the user, otherwise they are not shown. Not showing plot can be useful,
                when you want to edit the returned axis, before showing it.

            **figure_kwargs:
                Keyword arguments that are passed to the plt.figure, at its initialization.

        Returns:
            (plt.axis):
                Axis containing the performance plot.
        """
        plt.figure(**figure_kwargs)

        imp_methods = list(self.report_df["strategy"])
        performance = list(self.report_df[f"{self.scorer.metric_name} score"])
        std_error = list(self.report_df[f"{self.scorer.metric_name} std"])

        y_pos = [i for i, _ in enumerate(imp_methods)]
        x_spacing = 0.01
        y_spacing = 2 * x_spacing
        plt.barh(
            y_pos,
            performance,
            xerr=std_error,
            align="center",
            color=np.random.rand(len(performance), 3),
        )

        for index, value in enumerate(performance):
            plt.text(value + x_spacing, index + y_spacing, str(value), rotation=45)
        plt.yticks(y_pos, imp_methods)
        plt.xlabel(f"Metric ({(self.scorer.metric_name).replace('_',' ').upper()})")
        plt.title("Imputation Techniques")
        plt.margins(0.1)
        plt.tight_layout()
        ax = plt.gca()
        if show:
            plt.show()
        else:
            plt.close()
        return ax
