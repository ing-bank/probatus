# Copyright (c) 2020 ING Bank N.V.
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


from probatus.interpret import TreeDependencePlotter
from probatus.utils import assure_column_names_consistency, assure_pandas_df, shap_calc, assure_list_of_strings,\
    calculate_shap_importance, NotFittedError
from sklearn.metrics import roc_auc_score
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

class ShapModelInterpreter:
    """
    This class is a wrapper that allows to easily analyse model's features. It allows to plot SHAP feature importance,
     SHAP summary plot and SHAP dependence plots.

    Example:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    feature_names = ['f1', 'f2', 'f3', 'f4']

    # Prepare two samples
    X, y = make_classification(n_samples=1000, n_features=4, random_state=0)
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare and fit model. Remember about class_weight="balanced" or an equivalent.
    clf = RandomForestClassifier(class_weight='balanced', n_estimators = 100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Train ShapModelAnalyser
    shap_interpreter = ShapModelInterpreter(clf)
    feature_importance = shap_interpreter.fit_compute(X_train, X_test, y_train, y_test)

    # Make plots
    shap_interpreter.plot('importance')
    shap_interpreter.plot('summary')
    shap_interpreter.plot('dependence', target_columns=['f1', 'f2'])
    shap_interpreter.plot('sample', samples_index=[521, 78])
    ```
    """


    def __init__(self, clf):
        """
        Initializes the class.

        Args:
            clf (binary classifier): Model fitted on X_train.
        """
        self.clf = clf
        self.fitted = False


    def _check_if_fitted(self):
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))


    def fit(self, X_train, X_test, y_train, y_test, column_names=None, class_names=None, approximate=False,
            **shap_kwargs):
        """
        Fits the object and calculates the shap values for the provided datasets.

        Args:
            X_train (pd.DataFrame): Dataframe containing training data.
            X_test (pd.DataFrame): Dataframe containing test data.
            y_train (pd.Series): Series of binary labels for train data.
            y_test (pd.Series): Series of binary labels for test data.
            column_names (Optional, None, or list of str): List of feature names for the dataset. If None, then column
             names from the X_train dataframe are used.
            class_names (Optional, None, or list of str): List of class names e.g. ['neg', 'pos']. If none, the default
             ['Negative Class', 'Positive Class'] are used.
            approximate (boolean):, if True uses shap approximations - less accurate, but very fast.
            **shap_kwargs: keyword arguments passed to shap.TreeExplainer.
        """

        self.X_train = assure_pandas_df(X_train)
        self.X_test = assure_pandas_df(X_test)
        self.y_train = y_train
        self.y_test = y_test

        # Set class names
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = ['Negative Class', 'Positive Class']

        # Set column names
        self.column_names = assure_column_names_consistency(column_names, self.X_train)

        # Calculate Metrics
        self.auc_train = roc_auc_score(self.y_train, self.clf.predict_proba(self.X_train)[:, 1])
        self.auc_test = roc_auc_score(self.y_test, self.clf.predict_proba(self.X_test)[:, 1])
        self.results_text = "Train AUC: {},\nTest AUC: {}.".format(
            np.round(self.auc_train, 3),
            np.round(self.auc_test, 3)
        )

        self.shap_values, self.explainer = shap_calc(self.clf, self.X_test, approximate=approximate,
                                                     return_explainer=True, data=self.X_train, **shap_kwargs)

        # Get expected_value from the explainer
        self.expected_value = self.explainer.expected_value
        # For sklearn models the expected values consists of two elements (negative_class and positive_class)
        if isinstance(self.expected_value, list) or isinstance(self.expected_value, np.ndarray):
            self.expected_value = self.expected_value[1]

        # Initialize tree dependence plotter
        self.tdp = TreeDependencePlotter(self.clf).fit(self.X_test, self.y_test, precalc_shap=self.shap_values)

        self.fitted = True


    def compute(self):
        """
        Computes the DataFrame, that presents the importance of each feature.

        Returns:
            (pd.DataFrame): Dataframe with SHAP feature importance.
        """
        self._check_if_fitted()

        # Compute SHAP importance
        self.importance_df = calculate_shap_importance(self.shap_values, self.column_names)
        return self.importance_df


    def fit_compute(self,  X_train, X_test, y_train, y_test, column_names=None, class_names=None, approximate=False,
                    **shap_kwargs):
        """
        Fits the object and calculates the shap values for the provided datasets.

        Args:
            X_train (pd.DataFrame): Dataframe containing training data.
            X_test (pd.DataFrame): Dataframe containing test data.
            y_train (pd.Series): Series of binary labels for train data.
            y_test (pd.Series): Series of binary labels for test data.
            column_names (Optional, None, or list of str): List of feature names for the dataset. If None, then column
             names from the X_train dataframe are used.
            class_names (Optional, None, or list of str): List of class names e.g. ['neg', 'pos']. If none, the default
             ['Negative Class', 'Positive Class'] are used.
            approximate (boolean):, if True uses shap approximations - less accurate, but very fast.
            **shap_kwargs: keyword arguments passed to shap.TreeExplainer.

        Returns:
            (pd.DataFrame): Dataframe with SHAP feature importance.
        """
        self.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, column_names=column_names,
                 class_names=class_names, approximate=approximate, **shap_kwargs)
        return self.compute()


    def plot(self, plot_type, target_columns=None, samples_index=None, **plot_kwargs):
        """
        Plots the appropriate SHAP plot

        Args:
            plot_type (str): One of the following:

                - 'importance': Feature importance plot, SHAP bar summary plot
                - 'summary': SHAP Summary plot
                - 'dependence': Dependence plot for each feature
                - 'sample': Explanation of a given sample in the test data

            target_columns (Optional, None, str or list of str): List of features names, for which the plots should be
             generated. If None, all features will be plotted.

            samples_index (None, int, list or pd.Index): Index of samples to be explained if the `plot_type=sample`.

            **plot_kwargs: Keyword arguments passed to the plot method. For 'importance' and 'summary' plot_type, the
             kwargs are passed to shap.summary_plot, for 'dependence' plot_type, they are passed to
             probatus.interpret.TreeDependencePlotter.feature_plot method.

        """
        if target_columns is None:
            target_columns = self.column_names

        target_columns = assure_list_of_strings(target_columns, 'target_columns')
        target_columns_indices = [self.column_names.index(target_column) for target_column in target_columns]

        if plot_type in ['importance', 'summary']:
            # Get the target features
            target_X_test = self.X_test[target_columns]
            target_shap_values = self.shap_values[:, target_columns_indices]

            # Set summary plot settings
            if plot_type == 'importance':
                plot_type = 'bar'
                plot_title = 'SHAP Feature Importance'
            else:
                plot_type = 'dot'
                plot_title = 'SHAP Summary plot'

            shap.summary_plot(target_shap_values, target_X_test, plot_type=plot_type,
                              class_names=self.class_names, show=False, **plot_kwargs)
            ax = plt.gca()
            ax.set_title(plot_title)

            ax.annotate(self.results_text, (0, 0), (0, -50), fontsize=12, xycoords='axes fraction',
                        textcoords='offset points', va='top')
            plt.show()
        elif plot_type == 'dependence':
            ax = []
            for feature_name in target_columns:
                print()
                ax.append(
                    self.tdp.plot(feature=feature_name, figsize=(10, 7), target_names=self.class_names))
                plt.show()

        elif plot_type == 'sample':
            # Ensure the correct samples_index type
            if samples_index is None:
                raise(ValueError('For sample plot, you need to specify the samples_index be plotted plot'))
            elif isinstance(samples_index, int) or isinstance(samples_index, str):
                samples_index = [samples_index]
            elif not(isinstance(samples_index, list) or isinstance(samples_index, pd.Index)):
                raise(TypeError('sample_index must be one of the following: int, str, list or pd.Index'))

            ax = []
            for sample_index in samples_index:
                sample_loc = self.X_test.index.get_loc(sample_index)

                shap.plots._waterfall.waterfall_legacy(self.expected_value,
                                                       self.shap_values[sample_loc, :],
                                                       self.X_test.loc[sample_index],
                                                       show=False, **plot_kwargs)

                plot_title = f'SHAP Sample Explanation for index={sample_index}'
                current_ax = plt.gca()
                current_ax.set_title(plot_title)
                ax.append(current_ax)
                plt.show()
        else:
            raise ValueError("Wrong plot type, select from 'importance', 'summary', or 'dependence'")

        if isinstance(ax, list) and len(ax) == 1:
            ax = ax[0]
        return ax