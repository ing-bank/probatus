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
from probatus.utils import (
    preprocess_labels,
    get_single_scorer,
    preprocess_data,
    BaseFitComputePlotClass,
)
from probatus.utils.shap_helpers import shap_calc, calculate_shap_importance
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
import warnings


class BaseResemblanceModel(BaseFitComputePlotClass):
    """
    This model checks for the similarity of two samples.

    A possible use case is analysis of whether th train sample differs
    from the test sample, due to e.g. non-stationarity.

    This is a base class and needs to be extended by a fit() method, which implements how the data is split,
    how the model is trained and evaluated.
    Further, inheriting classes need to implement how feature importance should be indicated.
    """

    def __init__(
        self,
        clf,
        scoring="roc_auc",
        test_prc=0.25,
        n_jobs=1,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name aligned with
                predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric. The recommended option for this
                class is 'roc_auc'.

            test_prc (float, optional):
                Percentage of data used to test the model. By default 0.25 is set.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to an integer.
        """  # noqa
        self.clf = clf
        self.test_prc = test_prc
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.scorer = get_single_scorer(scoring)

    def _init_output_variables(self):
        """
        Initializes variables that will be filled in during fit() method, and are used as output.
        """
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_score = None
        self.test_score = None
        self.report = None

    def fit(self, X1, X2, column_names=None, class_names=None):
        """
        Base fit functionality that should be executed before each fit.

        Args:
            X1 (np.ndarray or pd.DataFrame):
                First sample to be compared. It needs to have the same number of columns as X2.

            X2 (np.ndarray or pd.DataFrame):
                Second sample to be compared. It needs to have the same number of columns as X1.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

            class_names (None, or list of str, optional):
                List of class names assigned, in this case provided samples e.g. ['sample1', 'sample2']. If none, the
                default ['First Sample', 'Second Sample'] are used.

        Returns:
            (BaseResemblanceModel):
                Fitted object
        """
        # Set seed for results reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Set class names
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = ["First Sample", "Second Sample"]

        # Ensure inputs are correct
        self.X1, self.column_names = preprocess_data(X1, X_name="X1", column_names=column_names, verbose=self.verbose)
        self.X2, _ = preprocess_data(X2, X_name="X2", column_names=column_names, verbose=self.verbose)

        # Prepare dataset for modelling
        self.X = pd.DataFrame(pd.concat([self.X1, self.X2], axis=0), columns=self.column_names).reset_index(drop=True)

        self.y = pd.Series(
            np.concatenate(
                [
                    np.zeros(self.X1.shape[0]),
                    np.ones(self.X2.shape[0]),
                ]
            )
        ).reset_index(drop=True)

        # Assure the type and number of classes for the variable
        self.X, _ = preprocess_data(self.X, X_name="X", column_names=self.column_names, verbose=self.verbose)

        self.y = preprocess_labels(self.y, y_name="y", index=self.X.index, verbose=self.verbose)

        # Reinitialize variables in case of multiple times being fit
        self._init_output_variables()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_prc,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.y,
        )
        self.clf.fit(self.X_train, self.y_train)

        self.train_score = np.round(self.scorer.score(self.clf, self.X_train, self.y_train), 3)
        self.test_score = np.round(self.scorer.score(self.clf, self.X_test, self.y_test), 3)

        self.results_text = (
            f"Train {self.scorer.metric_name}: {np.round(self.train_score, 3)},\n"
            f"Test {self.scorer.metric_name}: {np.round(self.test_score, 3)}."
        )
        if self.verbose > 50:
            print(f"Finished model training: \n{self.results_text}")

        if self.verbose > 0:
            if self.train_score > self.test_score:
                warnings.warn(
                    f"Train {self.scorer.metric_name} > Test {self.scorer.metric_name}, which might indicate "
                    f"an overfit. \n Strong overfit might lead to misleading conclusions when analysing "
                    f"feature importance. Consider retraining with more regularization applied to the model."
                )
        self.fitted = True
        return self

    def get_data_splits(self):
        """
        Returns the data splits used to train the Resemblance model.

        Returns:
            (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
                X_train, X_test, y_train, y_test.
        """
        self._check_if_fitted()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def compute(self, return_scores=False):
        """
        Checks if fit() method has been run and computes the output variables.

        Args:
            return_scores (bool, optional):
                Flag indicating whether the method should return a tuple (feature importances, train score,
                test score), or feature importances. By default the second option is selected.

        Returns:
            (tuple(pd.DataFrame, float, float) or pd.DataFrame):
                Depending on value of return_tuple either returns a tuple (feature importances, train AUC, test AUC), or
                feature importances.
        """
        self._check_if_fitted()

        if return_scores:
            return self.report, self.train_score, self.test_score
        else:
            return self.report

    def fit_compute(
        self,
        X1,
        X2,
        column_names=None,
        class_names=None,
        return_scores=False,
        **fit_kwargs,
    ):
        """
        Fits the resemblance model and computes the report regarding feature importance.

        Args:
            X1 (np.ndarray or pd.DataFrame):
                First sample to be compared. It needs to have the same number of columns as X2.

            X2 (np.ndarray or pd.DataFrame):
                Second sample to be compared. It needs to have the same number of columns as X1.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

            class_names (None, or list of str, optional):
                List of class names assigned, in this case provided samples e.g. ['sample1', 'sample2']. If none, the
                default ['First Sample', 'Second Sample'] are used.

            return_scores (bool, optional):
                Flag indicating whether the method should return a tuple (feature importances, train score,
                test score), or feature importances. By default the second option is selected.

            **fit_kwargs:
                In case any other arguments are accepted by fit() method, they can be passed as keyword arguments.

        Returns:
            (tuple of (pd.DataFrame, float, float) or pd.DataFrame):
                Depending on value of return_tuple either returns a tuple (feature importances, train AUC, test AUC), or
                feature importances.
        """
        self.fit(X1, X2, column_names=column_names, class_names=class_names, **fit_kwargs)
        return self.compute(return_scores=return_scores)

    def plot(self):
        """
        Plot.
        """
        raise (NotImplementedError("Plot method has not been implemented."))


class PermutationImportanceResemblance(BaseResemblanceModel):
    """
    This model checks the similarity of two samples.

    A possible use case is analysis of whether the train sample differs
    from the test sample, due to e.g. non-stationarity.

    It assigns labels to each sample, 0 to the first sample, 1 to the second. Then, it randomly selects a portion of
    data to train on. The resulting model tries to distinguish which sample a given test row comes from. This
    provides insights on how distinguishable these samples are and which features contribute to that. The feature
    importance is calculated using permutation importance.

    If the model achieves a test AUC significantly different than 0.5, it indicates that it is possible to distinguish
    between the samples, and therefore, the samples differ.
    Features with a high permutation importance contribute to that effect the most.
    Thus, their distribution might differ between two samples.

    Examples:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.sample_similarity import PermutationImportanceResemblance
    X1, _ = make_classification(n_samples=100, n_features=5)
    X2, _ = make_classification(n_samples=100, n_features=5, shift=0.5)
    clf = RandomForestClassifier(max_depth=2)
    perm = PermutationImportanceResemblance(clf)
    feature_importance = perm.fit_compute(X1, X2)
    perm.plot()
    ```
    <img src="../img/sample_similarity_permutation_importance.png" width="500" />
    """

    def __init__(
        self,
        clf,
        iterations=100,
        scoring="roc_auc",
        test_prc=0.25,
        n_jobs=1,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            iterations (int, optional):
                Number of iterations performed to calculate permutation importance. By default 100 iterations per
                feature are done.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name aligned with
                predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric. Recommended option for this
                class is 'roc_auc'.

            test_prc (float, optional):
                Percentage of data used to test the model. By default 0.25 is set.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """  # noqa
        super().__init__(
            clf=clf,
            scoring=scoring,
            test_prc=test_prc,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

        self.iterations = iterations

        self.iterations_columns = ["feature", "importance"]
        self.iterations_results = pd.DataFrame(columns=self.iterations_columns)

        self.plot_x_label = "Permutation Feature Importance"
        self.plot_y_label = "Feature Name"
        self.plot_title = "Permutation Feature Importance of Resemblance Model"

    def fit(self, X1, X2, column_names=None, class_names=None):
        """
        This function assigns labels to each sample, 0 to the first sample, 1 to the second.

        Then, it randomly selects a
            portion of data to train on. The resulting model tries to distinguish which sample a given test row
            comes from. This provides insights on how distinguishable these samples are and which features contribute to
            that. The feature importance is calculated using permutation importance.

        Args:
            X1 (np.ndarray or pd.DataFrame):
                First sample to be compared. It needs to have the same number of columns as X2.

            X2 (np.ndarray or pd.DataFrame):
                Second sample to be compared. It needs to have the same number of columns as X1.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

            class_names (None, or list of str, optional):
                List of class names assigned, in this case provided samples e.g. ['sample1', 'sample2']. If none, the
                default ['First Sample', 'Second Sample'] are used.

        Returns:
            (PermutationImportanceResemblance):
                Fitted object.
        """
        super().fit(X1=X1, X2=X2, column_names=column_names, class_names=class_names)

        permutation_result = permutation_importance(
            self.clf,
            self.X_test,
            self.y_test,
            scoring=self.scorer.scorer,
            n_repeats=self.iterations,
            n_jobs=self.n_jobs,
        )

        # Prepare report
        self.report_columns = ["mean_importance", "std_importance"]
        self.report = pd.DataFrame(index=self.column_names, columns=self.report_columns, dtype=float)

        for feature_index, feature_name in enumerate(self.column_names):
            # Fill in the report
            self.report.loc[feature_name, "mean_importance"] = permutation_result["importances_mean"][feature_index]
            self.report.loc[feature_name, "std_importance"] = permutation_result["importances_std"][feature_index]

            # Fill in the iterations
            current_iterations = pd.DataFrame(
                np.stack(
                    [
                        np.repeat(feature_name, self.iterations),
                        permutation_result["importances"][feature_index, :].reshape((self.iterations,)),
                    ],
                    axis=1,
                ),
                columns=self.iterations_columns,
            )

            self.iterations_results = pd.concat([self.iterations_results, current_iterations])

        self.iterations_results["importance"] = self.iterations_results["importance"].astype(float)

        # Sort by mean test score of first metric
        self.report.sort_values(by="mean_importance", ascending=False, inplace=True)

        return self

    def plot(self, ax=None, top_n=None, show=True, **plot_kwargs):
        """
        Plots the resulting AUC of the model as well as the feature importances.

        Args:
            ax (matplotlib.axes, optional):
                Axes to which the output should be plotted. If not provided new axes are created.

            top_n (int, optional):
                Number of the most important features to be plotted. By default features are included in the plot.

            show (bool, optional):
                If True, the plots are shown to the user, otherwise they are not shown. Not showing a plot can be useful
                when you want to edit the returned axis before showing it.

            **plot_kwargs:
                Keyword arguments passed to the matplotlib.plotly.subplots method.

        Returns:
            (matplotlib.axes):
                Axes that include the plot.
        """

        feature_report = self.compute()
        self.iterations_results["importance"] = self.iterations_results["importance"].astype(float)

        sorted_features = feature_report["mean_importance"].sort_values(ascending=True).index.values
        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[-top_n:]

        if ax is None:
            fig, ax = plt.subplots(**plot_kwargs)

        for position, feature in enumerate(sorted_features):
            ax.boxplot(
                self.iterations_results[self.iterations_results["feature"] == feature]["importance"],
                positions=[position],
                vert=False,
            )

        ax.set_yticks(range(position + 1))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel(self.plot_x_label)
        ax.set_ylabel(self.plot_y_label)
        ax.set_title(self.plot_title)

        ax.annotate(
            self.results_text,
            (0, 0),
            (0, -50),
            fontsize=12,
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

        if show:
            plt.show()
        else:
            plt.close()

        return ax


class SHAPImportanceResemblance(BaseResemblanceModel):
    """
    This model checks for similarity of two samples.

    A possible use case is analysis of whether the train sample differs
        from the test sample, due to e.g. non-stationarity.

    It assigns labels to each sample, 0 to the first sample, 1 to the second. Then, it randomly selects a portion of data
        to train on. The resulting model tries to distinguish which sample a given test row comes from. This
        provides insights on how distinguishable these samples are and which features contribute to that. The feature
        importance is calculated using SHAP feature importance.

    If the model achieves test AUC significantly different than 0.5, it indicates that it is possible to distinguish
    between the samples, and therefore, the samples differ. Features with a high permutation importance contribute to that
        effect the most. Thus, their distribution might differ between two samples.

    This class currently works only with the Tree based models.

    Examples:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.sample_similarity import SHAPImportanceResemblance
    X1, _ = make_classification(n_samples=100, n_features=5)
    X2, _ = make_classification(n_samples=100, n_features=5, shift=0.5)
    clf = RandomForestClassifier(max_depth=2)
    rm = SHAPImportanceResemblance(clf)
    feature_importance = rm.fit_compute(X1, X2)
    rm.plot()
    ```

    <img src="../img/sample_similarity_shap_importance.png" width="320" />
    <img src="../img/sample_similarity_shap_summary.png" width="320" />
    """

    def __init__(
        self,
        clf,
        scoring="roc_auc",
        test_prc=0.25,
        n_jobs=1,
        verbose=0,
        random_state=None,
    ):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name aligned with
                predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric. Recommended option for this
                class is 'roc_auc'.

            test_prc (float, optional):
                Percentage of data used to test the model. By default 0.25 is set.

            n_jobs (int, optional):
                Number of parallel executions. If -1 use all available cores. By default 1.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.
        """  # noqa
        super().__init__(
            clf=clf,
            scoring=scoring,
            test_prc=test_prc,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

        self.plot_title = "SHAP summary plot"

    def fit(self, X1, X2, column_names=None, class_names=None, **shap_kwargs):
        """
        This function assigns labels to each sample, 0 to the first sample, 1 to the second.

        Then, it randomly selects a
            portion of data to train on. The resulting model tries to distinguish which sample a given test row
            comes from. This provides insights on how distinguishable these samples are and which features contribute to
            that. The feature importance is calculated using SHAP feature importance.

        Args:
            X1 (np.ndarray or pd.DataFrame):
                First sample to be compared. It needs to have the same number of columns as X2.

            X2 (np.ndarray or pd.DataFrame):
                Second sample to be compared. It needs to have the same number of columns as X1.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

            class_names (None, or list of str, optional):
                List of class names assigned, in this case provided samples e.g. ['sample1', 'sample2']. If none, the
                default ['First Sample', 'Second Sample'] are used.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.

        Returns:
            (SHAPImportanceResemblance):
                Fitted object.
        """
        super().fit(X1=X1, X2=X2, column_names=column_names, class_names=class_names)

        self.shap_values_test = shap_calc(self.clf, self.X_test, verbose=self.verbose, **shap_kwargs)
        self.report = calculate_shap_importance(self.shap_values_test, self.column_names)
        return self

    def plot(self, plot_type="bar", show=True, **summary_plot_kwargs):
        """
        Plots the resulting AUC of the model as well as the feature importances.

        Args:
            plot_type (str, optional): Type of plot, used to compute shap.summary_plot. By default 'bar', available ones
                are  "dot", "bar", "violin",

            show (bool, optional):
                If True, the plots are showed to the user, otherwise they are not shown. Not showing plot can be useful,
                when you want to edit the returned axis, before showing it.

            **summary_plot_kwargs:
                kwargs passed to the shap.summary_plot.

        Returns:
            (matplotlib.axes):
                Axes that include the plot.
        """

        # This line serves as a double check if the object has been fitted
        self._check_if_fitted()

        shap.summary_plot(
            self.shap_values_test,
            self.X_test,
            plot_type=plot_type,
            class_names=self.class_names,
            show=False,
            **summary_plot_kwargs,
        )
        ax = plt.gca()
        ax.set_title(self.plot_title)

        ax.annotate(
            self.results_text,
            (0, 0),
            (0, -50),
            fontsize=12,
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

        if show:
            plt.show()
        else:
            plt.close()

        return ax

    def get_shap_values(self):
        """
        Gets the SHAP values generated on the test set.

        Returns:
             (np.array):
                SHAP values generated on the test set.
        """
        self._check_if_fitted()
        return self.shap_values_test
