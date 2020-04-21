import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from probatus.utils import assure_numpy_array, NotFittedError, get_scorers, warn_if_missing
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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

        # Init output variables
        self.report_columns = ['mean_importance', 'std_importance']
        self.iterations_columns = ['feature', 'importance']

        self.fitted = False

        self.metric_name = 'roc_auc'
        self.scorer = get_scorers(self.metric_name)[0]

        # Plot variables, can be overwritten by inheriting classes
        self.plot_x_label = 'Feature Importance'
        self.plot_y_label = 'Feature Name'
        self.plot_title = 'Predictive Power of Features'


    def init_output_variables(self):
        """
        Initializes variables that will be filled in during fit() method, and are used as output
        """
        self.iterations_results = pd.DataFrame(columns=self.iterations_columns)
        self.report = pd.DataFrame(index=self.columns, columns=self.report_columns, dtype=float)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.baseline_auc_train = None
        self.baseline_auc_test = None


    def fit(self, X1, X2, columns=None):
        """
        Base fit functionality that should be executed before each fit.

        Args:
            X1 (np.ndarray or pd.DataFrame): First sample to be compared. It needs to have the same number of columns
            as X2.

            X2 (np.ndarray or pd.DataFrame): Second sample to be compared. It needs to have the same number of columns
            as X1.

            columns (list of str, optional): List of feature names of the provided samples. If provided it will be used
            to overwrite the existing feature names. If not provided the existing feature names are used or default
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

        # Check if columns are passed correctly
        if columns is None:
            # Checking if original X1 was a df then taking its column names
            if isinstance(X1, pd.DataFrame):
                self.columns = X1.columns
            # Otherwise make own feature names
            else:
                self.columns = ['column_{}'.format(idx)  for idx in range(self.X1.shape[1])]
        else:
            if isinstance(columns, list):
                if  len(columns) == self.X1.shape[1]:
                    self.columns = columns
                else:
                    raise(ValueError("Passed columns have different dimensionality than input samples. "
                                     "The dimensionality of columns is {} and first sample {}".
                                     format(len(columns), self.X1.shape[1])))
            else:
                raise(TypeError("Passed columns must be a list"))

        # Prepare dataset for modelling
        self.X = pd.DataFrame(np.concatenate([
            self.X1,
            self.X2
        ]), columns = self.columns)

        self.y = pd.Series(np.concatenate([
            np.zeros(self.X1.shape[0]),
            np.ones(self.X2.shape[0]),
        ]))

        # Reinitialize variables in case of multiple times being fit
        self.init_output_variables()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,test_size=self.test_prc,
                                                                                random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

        self.baseline_auc_train = self.scorer.score(self.model, self.X_train, self.y_train)
        self.baseline_auc_test = self.scorer.score(self.model, self.X_test, self.y_test)
        self.fitted = True


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
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))

        # Ensure that importance column is float, otherwise plot might throw an error
        self.iterations_results['importance'] = self.iterations_results['importance'].astype(float)

        if return_auc:
            return self.report, self.baseline_auc_train, self.baseline_auc_test
        else:
            return self.report


    def fit_compute(self, X1, X2, columns=None, return_auc=False, **fit_kwargs):
        """
        Fits the resemblance model and computes the report regarding feature importance.

        Args:
            X1 (np.ndarray or pd.DataFrame): First sample to be compared. It needs to have the same number of columns
            as X2.

            X2 (np.ndarray or pd.DataFrame): Second sample to be compared. It needs to have the same number of columns
            as X1.

            columns (list of str, optional): List of feature names of the provided samples. If provided it will be used
            to overwrite the existing feature names. If not provided the existing feature names are used or default
            feature names are generated.

            return_auc (bool, optional): Flag indicating whether the method should return a tuple (feature
            importances, train AUC, test AUC), or only feature importances. By default the second option is selected.

            **fit_kwargs: arguments passed to the fit() method.

        Returns:
            tuple of (pd.DataFrame, float, float) or pd.DataFrame: Depending on value of return_tuple either returns a
            tuple (feature importances, train AUC, test AUC), or feature importances.
        """
        self.fit(X1, X2, columns=columns, **fit_kwargs)
        return self.compute(return_auc=return_auc)


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

        fig_text = "AUC performance of baseline model on train: {},\n" \
                   "AUC performance of baseline model on test: {}.". \
                       format(np.round(self.baseline_auc_train, 3), np.round(self.baseline_auc_test, 3))

        ax.annotate(fig_text, (0,0), (0, -50), fontsize=12, xycoords='axes fraction',
                    textcoords='offset points', va='top')

        return ax

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
        >>> X1, _ = make_classification(n_samples=900, n_features=5)
        >>> X2, _ = make_classification(n_samples=1000, n_features=5)
        >>> clf = RandomForestClassifier()
        >>> perm = PermutationResemblanceModel(clf)
        >>> feature_importance = perm.fit_compute(X1, X2)
        >>> perm.plot()
    """
    def __init__(self, model, iterations=100, **kwargs):
        super().__init__(model=model, **kwargs)
        self.iterations = iterations
        self.plot_x_label = 'Permutation Feature Importance'


    def fit(self, X1, X2, columns=None):
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

            columns (list of str, optional): List of feature names of the provided samples. If provided it will be used
            to overwrite the existing feature names. If not provided the existing feature names are used or default
            feature names are generated.
        """
        super().fit(X1=X1, X2=X2, columns=columns)

        permutation_result = permutation_importance(self.model, self.X_test, self.y_test, scoring=self.scorer.scorer,
                                                    n_repeats=self.iterations, n_jobs=self.n_jobs)

        for feature_index, feature_name in enumerate(self.columns):
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

