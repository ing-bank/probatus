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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from probatus.utils import assure_numpy_array, NotFittedError, get_scorers, warn_if_missing,\
    assure_column_names_consistency
from probatus.utils.shap_helpers import shap_calc, calculate_shap_importance
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
import warnings

class BaseResemblanceModel(object):
    """
    This model checks for similarity of two samples. A possible use case is analysis whether train sample differs
    from test sample, due to e.g. non-stationarity.

    This is a base class and needs to be extended by a fit() method, which implements how data is split, how model is
    trained and evaluated. Further, inheriting classes need to implement how feature importance should be indicated.

    Args:
        model (model object): Binary classification model or pipeline.

        test_prc (float, optional): Percentage of data used to test the model. By default 0.25 is set.

        n_jobs (int, optional): Number of parallel executions. If -1 use all available cores. By default 1.

        random_state (int, optional): The seed used by the random number generator.
    """
    def __init__(self, model, test_prc=0.25, n_jobs=1, random_state=42):
        self.model = model
        self.test_prc = test_prc
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.fitted = False

        self.metric_name = 'roc_auc'
        self.scorer = get_scorers(self.metric_name)[0]


    def init_output_variables(self):
        """
        Initializes variables that will be filled in during fit() method, and are used as output
        """
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.auc_train = None
        self.auc_test = None
        self.report = None


    def fit(self, X1, X2, column_names=None):
        """
        Base fit functionality that should be executed before each fit.

        Args:
            X1 (np.ndarray or pd.DataFrame): First sample to be compared. It needs to have the same number of columns
            as X2.

            X2 (np.ndarray or pd.DataFrame): Second sample to be compared. It needs to have the same number of columns
            as X1.

            column_names (list of str, optional): List of feature names of the provided samples. If provided it will be
            used to overwrite the existing feature names. If not provided the existing feature names are used or default
            feature names are generated.
        """

        # Set seed for results reproducibility
        np.random.seed(self.random_state)

        # Ensure inputs are correct
        self.X1 = assure_numpy_array(X1)
        self.X2 = assure_numpy_array(X2)

        # Check if any missing values
        warn_if_missing(self.X1, 'X1')
        warn_if_missing(self.X2, 'X2')

        # Ensure the same shapes
        if self.X1.shape[1] != self.X2.shape[1]:
            raise(ValueError("Passed variables do not have the same shape. The passed dimensions are {} and {}".
                             format(self.X1.shape[1], self.X2.shape[1])))

        # Check if column_names are passed correctly
        self.column_names = assure_column_names_consistency(column_names, X1)

        # Prepare dataset for modelling
        self.X = pd.DataFrame(np.concatenate([
            self.X1,
            self.X2
        ]), columns = self.column_names).reset_index(drop=True)

        self.y = pd.Series(np.concatenate([
            np.zeros(self.X1.shape[0]),
            np.ones(self.X2.shape[0]),
        ])).reset_index(drop=True)

        # Reinitialize variables in case of multiple times being fit
        self.init_output_variables()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,test_size=self.test_prc,
                                                                                random_state=self.random_state,
                                                                                stratify=self.y)
        self.model.fit(self.X_train, self.y_train)

        self.auc_train = np.round(self.scorer.score(self.model, self.X_train, self.y_train), 3)
        self.auc_test = np.round(self.scorer.score(self.model, self.X_test, self.y_test), 3)


        print(f'Finished model training: Train AUC {self.auc_train},'
              f' Test AUC {self.auc_test}')

        if self.auc_train > self.auc_test:
            warnings.warn('Train AUC > Test AUC, which might indicate an overfit. \n'
                          'Strong overfit might lead to misleading conclusions when analysing feature importance. '
                          'Consider retraining with more regularization applied to the model.')

        self.fitted = True


    def _check_if_fitted(self):
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))


    def get_data_splits(self):
        """
        Returns the data splits used to train the Resemblance model.

        Returns:
            (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series): X_train, X_test, y_train, y_test.
        """
        self._check_if_fitted()
        return self.X_train, self.X_test, self.y_train, self.y_test


    def compute(self, return_auc=False):
        """
        Checks if fit() method has been run and computes the output variables.

        Args:
            return_auc (bool, optional): Flag indicating whether the method should return a tuple (feature
            importances, train AUC, test AUC), or feature importances. By default the second option is selected.

        Returns:
            tuple(pd.DataFrame, float, float) or pd.DataFrame: Depending on value of return_tuple either returns a
            tuple (feature importances, train AUC, test AUC), or feature importances.
        """
        self._check_if_fitted()

        if return_auc:
            return self.report, self.auc_train, self.auc_test
        else:
            return self.report


    def fit_compute(self, X1, X2, column_names=None, return_auc=False, **fit_kwargs):
        """
        Fits the resemblance model and computes the report regarding feature importance.

        Args:
            X1 (np.ndarray or pd.DataFrame): First sample to be compared. It needs to have the same number of columns
            as X2.

            X2 (np.ndarray or pd.DataFrame): Second sample to be compared. It needs to have the same number of columns
            as X1.

            column_names (list of str, optional): List of feature names of the provided samples. If provided it will be
            used to overwrite the existing feature names. If not provided the existing feature names are used or default
            feature names are generated.

            return_auc (bool, optional): Flag indicating whether the method should return a tuple (feature
            importances, train AUC, test AUC), or only feature importances. By default the second option is selected.

            **fit_kwargs: arguments passed to the fit() method.

        Returns:
            tuple of (pd.DataFrame, float, float) or pd.DataFrame: Depending on value of return_tuple either returns a
            tuple (feature importances, train AUC, test AUC), or feature importances.
        """
        self.fit(X1, X2, column_names=column_names, **fit_kwargs)
        return self.compute(return_auc=return_auc)


class PermutationImportanceResemblance(BaseResemblanceModel):
    """
    This model checks for similarity of two samples. A possible use case is analysis whether train sample differs
    from test sample, due to e.g. non-stationarity.

    It assigns to labels to each sample, 0 to first sample, 1 to the second. Then, It randomly selects a portion of
    data to train on. The resulting model tries to distinguish which sample does a given test row comes from. This
    provides insights on how distinguishable these samples are and which features contribute to that. The feature
    importance is calculated using permutation importance.

    If the model achieves test AUC significantly different than 0.5, it indicates that it is possible to distinguish
    the samples, and therefore, the samples differ. Features with high permutation importance contribute to that
    effect the most. Thus, their distribution might differ between two samples.

    Args:
        model (model object): Binary classification model or pipeline.

        iterations (int, optional): Number of iterations performed to calculate permutation importance. By default 100
        iterations per feature are done.

        test_prc (float, optional): Percentage of data used to test the model. By default 0.25 is set.

        n_jobs (int, optional): Number of parallel executions. If -1 use all available cores. By default 1.

        random_state (int, optional): The seed used by the random number generator.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from probatus.sample_similarity import PermutationImportanceResemblance
        >>> X1, _ = make_classification(n_samples=1000, n_features=5)
        >>> X2, _ = make_classification(n_samples=1000, n_features=5, shift=0.5)
        >>> clf = RandomForestClassifier(max_depth=2)
        >>> perm = PermutationResemblanceModel(clf)
        >>> feature_importance = perm.fit_compute(X1, X2)
        >>> perm.plot()
    """

    def __init__(self, model, iterations=100, **kwargs):
        super().__init__(model=model, **kwargs)

        self.iterations = iterations

        self.iterations_columns = ['feature', 'importance']
        self.iterations_results = pd.DataFrame(columns=self.iterations_columns)

        self.plot_x_label = 'Permutation Feature Importance'
        self.plot_y_label = 'Feature Name'
        self.plot_title = 'Permutation Feature Importance of Resemblance Model'


    def fit(self, X1, X2, column_names=None):
        """
        This function assigns to labels to each sample, 0 to first sample, 1 to the second. Then, It randomly selects a
        portion of data to train on. The resulting model tries to distinguish which sample does a given test row comes
        from. This provides insights on how distinguishable these samples are and which features contribute to that. The
        feature importance is calculated using permutation importance.

        Args:
            X1 (np.ndarray or pd.DataFrame): First sample to be compared. It needs to have the same number of columns
            as X2.

            X2 (np.ndarray or pd.DataFrame): Second sample to be compared. It needs to have the same number of columns
            as X1.

            column_names (list of str, optional): List of feature names of the provided samples. If provided it will be
            used to overwrite the existing feature names. If not provided the existing feature names are used or default
            feature names are generated.
        """
        super().fit(X1=X1, X2=X2, column_names=column_names)


        permutation_result = permutation_importance(self.model, self.X_test, self.y_test, scoring=self.scorer.scorer,
                                                    n_repeats=self.iterations, n_jobs=self.n_jobs)

        # Prepare report
        self.report_columns = ['mean_importance', 'std_importance']
        self.report = pd.DataFrame(index=self.column_names, columns=self.report_columns, dtype=float)

        for feature_index, feature_name in enumerate(self.column_names):
            # Fill in the report
            self.report.loc[feature_name, 'mean_importance'] =\
                permutation_result['importances_mean'][feature_index]
            self.report.loc[feature_name, 'std_importance'] =\
                permutation_result['importances_std'][feature_index]

            # Fill in the iterations
            current_iterations = pd.DataFrame(
                np.stack([
                    np.repeat(feature_name, self.iterations),
                    permutation_result['importances'][feature_index, :].reshape((self.iterations,))
                    ], axis=1),
                columns=self.iterations_columns)

            self.iterations_results = pd.concat([self.iterations_results, current_iterations])

        self.iterations_results['importance'] = self.iterations_results['importance'].astype(float)

        # Sort by mean test score of first metric
        self.report.sort_values(by='mean_importance', ascending=False, inplace=True)


    def plot(self, ax=None, top_n=None):
        """
        Plots the resulting AUC of the model as well as the feature importances.

        Args:
            ax (matplotlib.axes, optional): Axes to which the output should be plotted. If not provided a new axes are
            created.

            top_n (int, optional): Number of the most important features to be plotted. By default are features are
            included into the plot.

        Returns:
            matplotlib.axes, optional: Axes that include the plot.
        """

        feature_report =  self.compute()
        self.iterations_results['importance'] =  self.iterations_results['importance'].astype(float)

        sorted_features = feature_report['mean_importance'].\
            sort_values(ascending=True).index.values
        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[-top_n:]

        if ax is None:
            height_per_subplot = len(sorted_features) / 2. + 1
            width_per_subplot = 10
            fig, ax = plt.subplots(figsize=(width_per_subplot, height_per_subplot))

        for position, feature in enumerate(sorted_features):
            ax.boxplot(self.iterations_results[self.iterations_results['feature']==feature]['importance'],
                       positions=[position], vert=False)

        ax.set_yticks(range(position + 1))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel(self.plot_x_label)
        ax.set_ylabel(self.plot_y_label)
        ax.set_title(self.plot_title)

        fig_text = "Train AUC: {},\n" \
                   "Test AUC: {}.". \
                       format(self.auc_train, self.auc_test)

        ax.annotate(fig_text, (0,0), (0, -50), fontsize=12, xycoords='axes fraction',
                    textcoords='offset points', va='top')

        return ax


class SHAPImportanceResemblance(BaseResemblanceModel):
    """
    This model checks for similarity of two samples. A possible use case is analysis whether train sample differs
    from test sample, due to e.g. non-stationarity.

    It assigns to labels to each sample, 0 to first sample, 1 to the second. Then, It randomly selects a portion of
    data to train on. The resulting model tries to distinguish which sample does a given test row comes from. This
    provides insights on how distinguishable these samples are and which features contribute to that. The feature
    importance is calculated using SHAP feature importance.

    If the model achieves test AUC significantly different than 0.5, it indicates that it is possible to distinguish
    the samples, and therefore, the samples differ. Features with high permutation importance contribute to that
    effect the most. Thus, their distribution might differ between two samples.

    This class currently works only with the Tree based models.

    Args:
        model (model object): Binary classification model or pipeline. It needs to be a tree based model, e.g.
        RandomForestClassifier, such that the shap.TreeExplainer can be used.

        test_prc (float, optional): Percentage of data used to test the model. By default 0.25 is set.

        n_jobs (int, optional): Number of parallel executions. If -1 use all available cores. By default 1.

        random_state (int, optional): The seed used by the random number generator.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from probatus.sample_similarity import SHAPImportanceResemblance
        >>> X1, _ = make_classification(n_samples=1000, n_features=5)
        >>> X2, _ = make_classification(n_samples=1000, n_features=5, shift=0.5)
        >>> clf = RandomForestClassifier(max_depth=2)
        >>> rm = SHAPImportanceResemblance(clf)
        >>> feature_importance = rm.fit_compute(X1, X2)
        >>> rm.plot()
    """

    def __init__(self, model,  **kwargs):
        super().__init__(model=model, **kwargs)

        self.plot_title = 'SHAP summary plot'

    def fit(self, X1, X2, column_names=None):
        """
        This function assigns to labels to each sample, 0 to first sample, 1 to the second. Then, It randomly selects a
        portion of data to train on. The resulting model tries to distinguish which sample does a given test row comes
        from. This provides insights on how distinguishable these samples are and which features contribute to that. The
        feature importance is calculated using SHAP feature importance.

        Args:
            X1 (np.ndarray or pd.DataFrame): First sample to be compared. It needs to have the same number of columns
            as X2.

            X2 (np.ndarray or pd.DataFrame): Second sample to be compared. It needs to have the same number of columns
            as X1.

            column_names (list of str, optional): List of feature names of the provided samples. If provided it will be
            used to overwrite the existing feature names. If not provided the existing feature names are used or default
            feature names are generated.
        """
        super().fit(X1=X1, X2=X2, column_names=column_names)

        self.shap_values_test = shap_calc(self.model, self.X_test, data=self.X_train)
        self.report = calculate_shap_importance(self.shap_values_test, self.column_names)

    def plot(self, plot_type='bar', **summary_plot_kwargs):
        """
        Plots the resulting AUC of the model as well as the feature importances.

        Args:
            plot_type (Optional, str): Type of plot, used to compute shap.summary_plot. By default 'bar', available ones
            are  "dot", "bar", "violin",

            **summary_plot_kwargs: kwargs passed to the shap.summary_plot

        Returns:
            matplotlib.axes, optional: Axes that include the plot.
        """

        # This line serves as a double check if the object has been fitted
        self._check_if_fitted()

        shap.summary_plot(self.shap_values_test, self.X_test, plot_type=plot_type,
                          class_names=['First Sample', 'Second Sample'], show=False, **summary_plot_kwargs)
        ax = plt.gca()
        ax.set_title(self.plot_title)

        fig_text = "Train AUC: {},\n" \
                   "Test AUC: {}.". \
                       format(self.auc_train, self.auc_test)

        ax.annotate(fig_text, (0,0), (0, -50), fontsize=12, xycoords='axes fraction',
                    textcoords='offset points', va='top')
        return ax


    def get_shap_values(self):
        '''
        Gets the SHAP values generated on the test set.

        Returns:
             (np.array) SHAP values generated on the test set.
        '''
        self._check_if_fitted()
        return self.shap_values_test
