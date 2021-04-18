import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from probatus.utils import (
    BaseFitComputePlotClass,
    assure_pandas_series,
    calculate_shap_importance,
    get_single_scorer,
    preprocess_data,
    preprocess_labels,
    shap_calc,
)
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV


class ShapRFECV(BaseFitComputePlotClass):
    """
    This class performs Backwards Recursive Feature Elimination, using SHAP feature importance.

    At each round, for a
        given feature set, starting from all available features, the following steps are applied:

    1. (Optional) Tune the hyperparameters of the model using sklearn compatible search CV e.g.
        [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html),
        [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomized#sklearn.model_selection.RandomizedSearchCV), or
        [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html),
    2. Apply Cross-validation (CV) to estimate the SHAP feature importance on the provided dataset. In each CV
        iteration, the model is fitted on the train folds, and applied on the validation fold to estimate
        SHAP feature importance.
    3. Remove `step` lowest SHAP importance features from the dataset.

    At the end of the process, the user can plot the performance of the model for each iteration, and select the
        optimal number of features and the features set.

    The functionality is
        similar to [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).
        The main difference is removing the lowest importance features based on SHAP features importance. It also
        supports the use of sklearn compatible search CV for hyperparameter optimization e.g.
        [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html),
        [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomized#sklearn.model_selection.RandomizedSearchCV), or
        [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html), which
        needs to be passed as the `clf`. Thanks to this you can perform hyperparameter optimization at each step of
        the feature elimination. Lastly, it supports categorical features (object and category dtype) and missing values
        in the data, as long as the model supports them.

    We recommend using [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
        because by default it handles missing values and categorical features. In case of other models, make sure to
        handle these issues for your dataset and consider impact it might have on features importance.


    Example:
    ```python
    import numpy as np
    import pandas as pd
    from probatus.feature_elimination import ShapRFECV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    feature_names = [
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
        'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
        'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20']

    # Prepare two samples
    X, y = make_classification(n_samples=200, class_sep=0.05, n_informative=6, n_features=20,
                               random_state=0, n_redundant=10, n_clusters_per_class=1)
    X = pd.DataFrame(X, columns=feature_names)


    # Prepare model and parameter search space
    clf = RandomForestClassifier(max_depth=5, class_weight='balanced')

    param_grid = {
        'n_estimators': [5, 7, 10],
        'min_samples_leaf': [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(clf, param_grid)


    # Run feature elimination
    shap_elimination = ShapRFECV(
        clf=search, step=0.2, cv=10, scoring='roc_auc', n_jobs=3)
    report = shap_elimination.fit_compute(X, y)

    # Make plots
    performance_plot = shap_elimination.plot()

    # Get final feature set
    final_features_set = shap_elimination.get_reduced_features_set(num_features=3)
    ```
    <img src="../img/shaprfecv.png" width="500" />

    """  # noqa

    def __init__(
        self,
        clf,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        random_state=None,
    ):
        """
        This method initializes the class.

        Args:
            clf (binary classifier, sklearn compatible search CV e.g. GridSearchCV, RandomizedSearchCV or BayesSearchCV):
                A model that will be optimized and trained at each round of feature elimination. The recommended model
                is [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
                because it by default handles the missing values and categorical variables. This parameter also supports
                any hyperparameter search schema that is consistent with the sklearn API e.g.
                [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
                [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
                or [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html#skopt.BayesSearchCV).

            step (int or float, optional):
                Number of lowest importance features removed each round. If it is an int, then each round such a number of
                features are discarded. If float, such a percentage of remaining features (rounded down) is removed each
                iteration. It is recommended to use float, since it is faster for a large number of features, and slows
                down and becomes more precise with fewer features. Note: the last round may remove fewer features in
                order to reach min_features_to_select.
                If columns_to_keep parameter is specified in the fit method, step is the number of features to remove after
                keeping those columns.

            min_features_to_select (int, optional):
                Minimum number of features to be kept. This is a stopping criterion of the feature elimination. By
                default the process stops when one feature is left. If columns_to_keep is specified in the fit method,
                it may overide this parameter to the maximum between length of columns_to_keep the two.

            cv (int, cross-validation generator or an iterable, optional):
                Determines the cross-validation splitting strategy. Compatible with sklearn
                [cv parameter](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).
                If None, then cv of 5 is used.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name aligned with predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric.

            n_jobs (int, optional):
                Number of cores to run in parallel while fitting across folds. None means 1 unless in a
                `joblib.parallel_backend` context. -1 means using all processors.

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
        if isinstance(self.clf, BaseSearchCV):
            self.search_clf = True
        else:
            self.search_clf = False

        if (isinstance(step, int) or isinstance(step, float)) and step > 0:
            self.step = step
        else:
            raise (
                ValueError(
                    f"The current value of step = {step} is not allowed. "
                    f"It needs to be a positive integer or positive float."
                )
            )

        if isinstance(min_features_to_select, int) and min_features_to_select > 0:
            self.min_features_to_select = min_features_to_select
        else:
            raise (
                ValueError(
                    f"The current value of min_features_to_select = {min_features_to_select} is not allowed. "
                    f"It needs to be a greater than or equal to 0."
                )
            )

        self.cv = cv
        self.scorer = get_single_scorer(scoring)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.report_df = pd.DataFrame([])
        self.verbose = verbose

    def _get_current_features_to_remove(self, shap_importance_df, columns_to_keep=None):
        """
        Implements the logic used to determine which features to remove.

        If step is a positive integer,
            at each round step lowest SHAP importance features are selected. If it is a float, such percentage
            of remaining features (rounded up) is removed each iteration. It is recommended to use float, since it is
            faster for a large set of features, and slows down and becomes more precise with fewer features.

        Args:
            shap_importance_df (pd.DataFrame):
                DataFrame presenting SHAP importance of remaining features.

        Returns:
            (list):
                List of features to be removed at a given round.
        """

        # Bounding the variable.
        num_features_to_remove = 0

        # If columns_to_keep is not None, exclude those columns and
        # calculate features to remove.
        if columns_to_keep is not None:
            mask = shap_importance_df.index.isin(columns_to_keep)
            shap_importance_df = shap_importance_df[~mask]

        # If the step is an int remove n features.
        if isinstance(self.step, int):
            num_features_to_remove = self._calculate_number_of_features_to_remove(
                current_num_of_features=shap_importance_df.shape[0],
                num_features_to_remove=self.step,
                min_num_features_to_keep=self.min_features_to_select,
            )
        # If the step is a float remove n * number features that are left, rounded down
        elif isinstance(self.step, float):
            current_step = int(np.floor(shap_importance_df.shape[0] * self.step))
            # The step after rounding down should be at least 1
            if current_step < 1:
                current_step = 1

            num_features_to_remove = self._calculate_number_of_features_to_remove(
                current_num_of_features=shap_importance_df.shape[0],
                num_features_to_remove=current_step,
                min_num_features_to_keep=self.min_features_to_select,
            )

        if num_features_to_remove == 0:
            return []
        else:
            return shap_importance_df.iloc[-num_features_to_remove:].index.tolist()

    @staticmethod
    def _calculate_number_of_features_to_remove(
        current_num_of_features,
        num_features_to_remove,
        min_num_features_to_keep,
    ):
        """
        Calculates the number of features to be removed.

        Makes sure that after removal at least
            min_num_features_to_keep are kept

         Args:
            current_num_of_features (int):
                Current number of features in the data.

            num_features_to_remove (int):
                Number of features to be removed at this stage.

            min_num_features_to_keep (int):
                Minimum number of features to be left after removal.

        Returns:
            (int):
                Number of features to be removed.
        """
        num_features_after_removal = current_num_of_features - num_features_to_remove
        if num_features_after_removal >= min_num_features_to_keep:
            num_to_remove = num_features_to_remove
        else:
            # take all available features minus number of them that should stay
            num_to_remove = current_num_of_features - min_num_features_to_keep
        return num_to_remove

    def _report_current_results(
        self,
        round_number,
        current_features_set,
        features_to_remove,
        train_metric_mean,
        train_metric_std,
        val_metric_mean,
        val_metric_std,
    ):
        """
        This function adds the results from a current iteration to the report.

        Args:
            round_number (int):
                Current number of the round.

            current_features_set (list of str):
                Current list of features.

            features_to_remove (list of str):
                List of features to be removed at the end of this iteration.

            train_metric_mean (float or int):
                Mean scoring metric measured on train set during CV.

            train_metric_std (float or int):
                Std scoring metric measured on train set during CV.

            val_metric_mean (float or int):
                Mean scoring metric measured on validation set during CV.

            val_metric_std (float or int):
                Std scoring metric measured on validation set during CV.
        """

        current_results = {
            "num_features": len(current_features_set),
            "features_set": None,
            "eliminated_features": None,
            "train_metric_mean": train_metric_mean,
            "train_metric_std": train_metric_std,
            "val_metric_mean": val_metric_mean,
            "val_metric_std": val_metric_std,
        }

        current_row = pd.DataFrame(current_results, index=[round_number])
        current_row["features_set"] = [current_features_set]
        current_row["eliminated_features"] = [features_to_remove]

        self.report_df = pd.concat([self.report_df, current_row], axis=0)

    def _get_feature_shap_values_per_fold(
        self,
        X,
        y,
        clf,
        train_index,
        val_index,
        sample_weight=None,
        **shap_kwargs,
    ):
        """
        This function calculates the shap values on validation set, and Train and Val score.

        Args:
            X (pd.DataFrame):
                Dataset used in CV.

            y (pd.Series):
                Binary labels for X.

            clf (binary classifier):
                Model to be fitted on the train folds.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.
        Returns:
            (np.array, float, float):
                Tuple with the results: Shap Values on validation fold, train score, validation score.
        """
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if sample_weight is not None:
            clf = clf.fit(X_train, y_train, sample_weight=sample_weight.iloc[train_index])
        else:
            clf = clf.fit(X_train, y_train)

        # Score the model
        score_train = self.scorer.scorer(clf, X_train, y_train)
        score_val = self.scorer.scorer(clf, X_val, y_val)

        # Compute SHAP values
        shap_values = shap_calc(clf, X_val, verbose=self.verbose, **shap_kwargs)
        return shap_values, score_train, score_val

    def fit(self, X, y, sample_weight=None, columns_to_keep=None, column_names=None, **shap_kwargs):
        """
        Fits the object with the provided data.

        The algorithm starts with the entire dataset, and then sequentially
             eliminates features. If sklearn compatible search CV is passed as clf e.g.
             [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
             [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
             or [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html),
             the hyperparameter optimization is applied at each step of the elimination.
             Then, the SHAP feature importance is calculated using Cross-Validation,
             and `step` lowest importance features are removed.

        Args:
            X (pd.DataFrame):
                Provided dataset.

            y (pd.Series):
                Binary labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            columns_to_keep (list of str, optional):
                List of column names to keep. If given,
                these columns will not be eliminated by the feature elimination process.
                However, these feature will used for the calculation of the SHAP values.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.

        Returns:
            (ShapRFECV): Fitted object.
        """
        # Set seed for results reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # If to columns_to_keep is not provided, then initialise it by an empty string.
        # If provided check if all the elements in columns_to_keep are of type string.
        if columns_to_keep is None:
            len_columns_to_keep = 0
        else:
            if all(isinstance(x, str) for x in columns_to_keep):
                len_columns_to_keep = len(columns_to_keep)
            else:
                raise (
                    ValueError(
                        "The current values of columns_to_keep are not allowed.All the elements should be strings."
                    )
                )

        # If the columns_to_keep parameter is provided, check if they match the column names in the X.
        if column_names is not None:
            if all(x in column_names for x in list(X.columns)):
                pass
            else:
                raise (ValueError("The column names in parameter columns_to_keep and column_names are not macthing."))

        # Check that the total number of columns to select is less than total number of columns in the data.
        # only when both parameters are provided.
        if column_names is not None and columns_to_keep is not None:
            if (self.min_features_to_select + len_columns_to_keep) > len(self.column_names):
                raise ValueError(
                    "Minimum features to select is greater than number of features."
                    "Lower the value for min_features_to_select or number of columns in columns_to_keep"
                )

        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, y_name="y", index=self.X.index, verbose=self.verbose)
        if sample_weight is not None:
            if self.verbose > 0:
                warnings.warn(
                    "sample_weight is passed only to the fit method of the model, not the evaluation metrics."
                )
            sample_weight = assure_pandas_series(sample_weight, index=self.X.index)
        self.cv = check_cv(self.cv, self.y, classifier=is_classifier(self.clf))

        remaining_features = current_features_set = self.column_names
        round_number = 0

        # Stop when stopping criteria is met.
        stopping_criteria = np.max([self.min_features_to_select, len_columns_to_keep])

        # Setting up the min_features_to_select parameter.
        if columns_to_keep is None:
            pass
        else:
            self.min_features_to_select = 0
            # This ensures that, if columns_to_keep is provided ,
            # the last features remaining are only the columns_to_keep.
            if self.verbose > 50:
                warnings.warn(f"Minimum features to select : {stopping_criteria}")

        while len(current_features_set) > stopping_criteria:
            round_number += 1

            # Get current dataset info
            current_features_set = remaining_features
            if columns_to_keep is None:
                remaining_removeable_features = list(set(current_features_set))
            else:
                remaining_removeable_features = list(set(current_features_set) | set(columns_to_keep))
            current_X = self.X[remaining_removeable_features]

            # Set seed for results reproducibility
            if self.random_state is not None:
                np.random.seed(self.random_state)

            # Optimize parameters
            if self.search_clf:
                current_search_clf = clone(self.clf).fit(current_X, self.y)
                current_clf = current_search_clf.estimator.set_params(**current_search_clf.best_params_)
            else:
                current_clf = clone(self.clf)

            # Perform CV to estimate feature importance with SHAP
            results_per_fold = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_feature_shap_values_per_fold)(
                    X=current_X,
                    y=self.y,
                    clf=current_clf,
                    train_index=train_index,
                    val_index=val_index,
                    sample_weight=sample_weight,
                    **shap_kwargs,
                )
                for train_index, val_index in self.cv.split(current_X, self.y)
            )

            shap_values = np.vstack([current_result[0] for current_result in results_per_fold])
            scores_train = [current_result[1] for current_result in results_per_fold]
            scores_val = [current_result[2] for current_result in results_per_fold]

            # Calculate the shap features with remaining features and features to keep.

            shap_importance_df = calculate_shap_importance(shap_values, remaining_removeable_features)

            # Get features to remove
            features_to_remove = self._get_current_features_to_remove(
                shap_importance_df, columns_to_keep=columns_to_keep
            )
            remaining_features = list(set(current_features_set) - set(features_to_remove))

            # Report results
            self._report_current_results(
                round_number=round_number,
                current_features_set=current_features_set,
                features_to_remove=features_to_remove,
                train_metric_mean=np.round(np.mean(scores_train), 3),
                train_metric_std=np.round(np.std(scores_train), 3),
                val_metric_mean=np.round(np.mean(scores_val), 3),
                val_metric_std=np.round(np.std(scores_val), 3),
            )
            if self.verbose > 50:
                print(
                    f"Round: {round_number}, Current number of features: {len(current_features_set)}, "
                    f'Current performance: Train {self.report_df.loc[round_number]["train_metric_mean"]} '
                    f'+/- {self.report_df.loc[round_number]["train_metric_std"]}, CV Validation '
                    f'{self.report_df.loc[round_number]["val_metric_mean"]} '
                    f'+/- {self.report_df.loc[round_number]["val_metric_std"]}. \n'
                    f"Features left: {remaining_features}. "
                    f"Removed features at the end of the round: {features_to_remove}"
                )
        self.fitted = True
        return self

    def compute(self):
        """
        Checks if fit() method has been run.

        and computes the DataFrame with results of feature elimintation for each round.

        Returns:
            (pd.DataFrame):
                DataFrame with results of feature elimination for each round.
        """
        self._check_if_fitted()

        return self.report_df

    def fit_compute(self, X, y, sample_weight=None, columns_to_keep=None, column_names=None, **shap_kwargs):
        """
        Fits the object with the provided data.

        The algorithm starts with the entire dataset, and then sequentially
             eliminates features. If sklearn compatible search CV is passed as clf e.g.
             [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
             [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
             or [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html),
             the hyperparameter optimization is applied at each step of the elimination.
             Then, the SHAP feature importance is calculated using Cross-Validation,
             and `step` lowest importance features are removed. At the end, the
             report containing results from each iteration is computed and returned to the user.

        Args:
            X (pd.DataFrame):
                Provided dataset.

            y (pd.Series):
                Binary labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            columns_to_keep (list of str, optional):
                List of columns to keep. If given, these columns will not be eliminated.

            column_names (list of str, optional):
                List of feature names of the provided samples. If provided it will be used to overwrite the existing
                feature names. If not provided the existing feature names are used or default feature names are
                generated.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.

        Returns:
            (pd.DataFrame):
                DataFrame containing results of feature elimination from each iteration.
        """

        self.fit(
            X,
            y,
            sample_weight=sample_weight,
            columns_to_keep=columns_to_keep,
            column_names=column_names,
            **shap_kwargs,
        )
        return self.compute()

    def get_reduced_features_set(self, num_features):
        """
        Gets the features set after the feature elimination process, for a given number of features.

        Args:
            num_features (int):
                Number of features in the reduced features set.

        Returns:
            (list of str):
                Reduced features set.
        """
        self._check_if_fitted()

        if num_features not in self.report_df.num_features.tolist():
            raise (
                ValueError(
                    f"The provided number of features has not been achieved at any stage of the process. "
                    f"You can select one of the following: {self.report_df.num_features.tolist()}"
                )
            )
        else:
            return self.report_df[self.report_df.num_features == num_features]["features_set"].values[0]

    def plot(self, show=True, **figure_kwargs):
        """
        Generates plot of the model performance for each iteration of feature elimination.

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
        x_ticks = list(reversed(self.report_df["num_features"].tolist()))

        plt.figure(**figure_kwargs)

        plt.plot(
            self.report_df["num_features"],
            self.report_df["train_metric_mean"],
            label="Train Score",
        )
        plt.fill_between(
            pd.to_numeric(self.report_df.num_features, errors="coerce"),
            self.report_df["train_metric_mean"] - self.report_df["train_metric_std"],
            self.report_df["train_metric_mean"] + self.report_df["train_metric_std"],
            alpha=0.3,
        )

        plt.plot(
            self.report_df["num_features"],
            self.report_df["val_metric_mean"],
            label="Validation Score",
        )
        plt.fill_between(
            pd.to_numeric(self.report_df.num_features, errors="coerce"),
            self.report_df["val_metric_mean"] - self.report_df["val_metric_std"],
            self.report_df["val_metric_mean"] + self.report_df["val_metric_std"],
            alpha=0.3,
        )

        plt.xlabel("Number of features")
        plt.ylabel(f"Performance {self.scorer.metric_name}")
        plt.title("Backwards Feature Elimination using SHAP & CV")
        plt.legend(loc="lower left")
        ax = plt.gca()
        ax.invert_xaxis()
        ax.set_xticks(x_ticks)
        if show:
            plt.show()
        else:
            plt.close()
        return ax


class EarlyStoppingShapRFECV(ShapRFECV):
    """
    This class performs Backwards Recursive Feature Elimination, using SHAP feature importance.

    This is a child of ShapRFECV which allows early stopping of the training step, available in models such as
        XGBoost and LightGBM. If you are not using early stopping, you should use the parent class,
        ShapRFECV, instead of EarlyStoppingShapRFECV.

    [Early stopping](https://en.wikipedia.org/wiki/Early_stopping) is a type of
        regularization technique in which the model is trained until the scoring metric, measured on a validation set,
        stops improving after a number of early_stopping_rounds. In boosted tree models, this technique can increase
        the training speed, by skipping the training of trees that do not improve the scoring metric any further,
        which is particularly useful when the training dataset is large.

    Note that if the classifier is a hyperparameter search model is used, the early stopping parameter is passed only
        to the fit method of the model duiring the Shapley values estimation step, and not for the hyperparameter
        search step.
        Early stopping can be seen as a type of regularization of the optimal number of trees. Therefore you can use
        it directly with a LightGBM or XGBoost model, as an alternative to a hyperparameter search model.

    At each round, for a
        given feature set, starting from all available features, the following steps are applied:

    1. (Optional) Tune the hyperparameters of the model using sklearn compatible search CV e.g.
        [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html),
        [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomized#sklearn.model_selection.RandomizedSearchCV), or
        [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html).
        Note that during this step the model does not use early stopping.
    2. Apply Cross-validation (CV) to estimate the SHAP feature importance on the provided dataset. In each CV
        iteration, the model is fitted on the train folds, and applied on the validation fold to estimate
        SHAP feature importance. The model is trained until the scoring metric eval_metric, measured on the
        validation fold, stops improving after a number of early_stopping_rounds.
    3. Remove `step` lowest SHAP importance features from the dataset.

    At the end of the process, the user can plot the performance of the model for each iteration, and select the
        optimal number of features and the features set.

    We recommend using [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
        because by default it handles missing values and categorical features. In case of other models, make sure to
        handle these issues for your dataset and consider impact it might have on features importance.


    Example:
    ```python
    from lightgbm import LGBMClassifier
    import pandas as pd
    from probatus.feature_elimination import EarlyStoppingShapRFECV
    from sklearn.datasets import make_classification

    feature_names = [
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
        'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
        'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20']

    # Prepare two samples
    X, y = make_classification(n_samples=200, class_sep=0.05, n_informative=6, n_features=20,
                               random_state=0, n_redundant=10, n_clusters_per_class=1)
    X = pd.DataFrame(X, columns=feature_names)

    # Prepare model
    clf = LGBMClassifier(n_estimators=200, max_depth=3)

    # Run feature elimination
    shap_elimination = EarlyStoppingShapRFECV(
        clf=clf, step=0.2, cv=10, scoring='roc_auc', early_stopping_rounds=10, n_jobs=3)
    report = shap_elimination.fit_compute(X, y)

    # Make plots
    performance_plot = shap_elimination.plot()

    # Get final feature set
    final_features_set = shap_elimination.get_reduced_features_set(num_features=3)
    ```
    <img src="../img/earlystoppingshaprfecv.png" width="500" />

    """  # noqa

    def __init__(
        self,
        clf,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        random_state=None,
        early_stopping_rounds=5,
        eval_metric="auc",
    ):
        """
        This method initializes the class.

        Args:
            clf (binary classifier, sklearn compatible search CV e.g. GridSearchCV, RandomizedSearchCV or BayesSearchCV):
                A model that will be optimized and trained at each round of features elimination. The model must
                support early stopping of training, which is the case for XGBoost and LightGBM, for example. The
                recommended model is [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
                because it by default handles the missing values and categorical variables. This parameter also supports
                any hyperparameter search schema that is consistent with the sklearn API e.g.
                [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
                [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
                or [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html#skopt.BayesSearchCV).
                Note that if a hyperparemeter search model is used, the hyperparameters are tuned without early
                stopping. Early stopping is applied only during the Shapley values estimation for feature
                elimination. We recommend simply passing the model without hyperparameter optimization, or using
                ShapRFECV without early stopping.


            step (int or float, optional):
                Number of lowest importance features removed each round. If it is an int, then each round such number of
                features is discarded. If float, such percentage of remaining features (rounded down) is removed each
                iteration. It is recommended to use float, since it is faster for a large number of features, and slows
                down and becomes more precise towards less features. Note: the last round may remove fewer features in
                order to reach min_features_to_select.
                If columns_to_keep parameter is specified in the fit method, step is the number of features to remove after
                keeping those columns.

            min_features_to_select (int, optional):
                Minimum number of features to be kept. This is a stopping criterion of the feature elimination. By
                default the process stops when one feature is left. If columns_to_keep is specified in the fit method,
                it may overide this parameter to the maximum between length of columns_to_keep the two.

            cv (int, cross-validation generator or an iterable, optional):
                Determines the cross-validation splitting strategy. Compatible with sklearn
                [cv parameter](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).
                If None, then cv of 5 is used.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name  aligned with predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric.

            n_jobs (int, optional):
                Number of cores to run in parallel while fitting across folds. None means 1 unless in a
                `joblib.parallel_backend` context. -1 means using all processors.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - nether prints nor warnings are shown
                - 1 - 50 - only most important warnings
                - 51 - 100 - shows other warnings and prints
                - above 100 - presents all prints and all warnings (including SHAP warnings).

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.

            early_stopping_rounds (int, optional):
                Number of rounds with constant performance after which the model fitting stops. This is passed to the
                fit method of the model for Shapley values estimation, but not for hyperparameter search. Only
                supported by some models, such as XGBoost and LightGBM.

            eval_metric (str, optional):
                Metric for scoring fitting rounds and activating early stopping. This is passed to the
                fit method of the model for Shapley values estimation, but not for hyperparameter search. Only
                supported by some models, such as [XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
                 and [LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters).
                Note that `eval_metric` is an argument of the model's fit method and it is different from `scoring`.
        """  # noqa
        super(EarlyStoppingShapRFECV, self).__init__(
            clf,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        if self.search_clf:
            if self.verbose > 0:
                warnings.warn(
                    "Early stopping will be used only during Shapley value estimation step, and not for hyperparameter"
                    "optimization."
                )

        if isinstance(early_stopping_rounds, int) and early_stopping_rounds > 0:
            self.early_stopping_rounds = early_stopping_rounds
        else:
            raise (
                ValueError(
                    f"The current value of early_stopping_rounds = {early_stopping_rounds} is not allowed. "
                    f"It needs to be a positive integer."
                )
            )

        self.eval_metric = eval_metric

    def _get_feature_shap_values_per_fold(self, X, y, clf, train_index, val_index, sample_weight=None, **shap_kwargs):
        """
        This function calculates the shap values on validation set, and Train and Val score.

        Args:
            X (pd.DataFrame):
                Dataset used in CV.

            y (pd.Series):
                Binary labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            clf (binary classifier):
                Model to be fitted on the train folds.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.
        Returns:
            (np.array, float, float):
                Tuple with the results: Shap Values on validation fold, train score, validation score.
        """
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if sample_weight is not None:
            clf = clf.fit(
                X_train,
                y_train,
                sample_weight=sample_weight.iloc[train_index],
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.eval_metric,
            )
        else:
            clf = clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.eval_metric,
            )
        # Score the model
        score_train = self.scorer.scorer(clf, X_train, y_train)
        score_val = self.scorer.scorer(clf, X_val, y_val)

        # Compute SHAP values
        shap_values = shap_calc(clf, X_val, verbose=self.verbose, **shap_kwargs)
        return shap_values, score_train, score_val
