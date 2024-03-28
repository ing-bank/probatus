import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV

from probatus.utils import (
    BaseFitComputePlotClass,
    assure_pandas_series,
    calculate_shap_importance,
    preprocess_data,
    preprocess_labels,
    get_single_scorer,
    shap_calc,
)


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
        needs to be passed as the `model`. Thanks to this you can perform hyperparameter optimization at each step of
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
    model = RandomForestClassifier(max_depth=5, class_weight='balanced')

    param_grid = {
        'n_estimators': [5, 7, 10],
        'min_samples_leaf': [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(model, param_grid)


    # Run feature elimination
    shap_elimination = ShapRFECV(
        model=search, step=0.2, cv=10, scoring='roc_auc', n_jobs=3)
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
        model,
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
            model (classifier or regressor, sklearn compatible search CV e.g. GridSearchCV, RandomizedSearchCV or BayesSearchCV):
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
                it may override this parameter to the maximum between length of columns_to_keep the two.

            cv (int, cross-validation generator or an iterable, optional):
                Determines the cross-validation splitting strategy. Compatible with sklearn
                [cv parameter](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).
                If None, then cv of 5 is used.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name aligned with predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).

            n_jobs (int, optional):
                Number of cores to run in parallel while fitting across folds. None means 1 unless in a
                `joblib.parallel_backend` context. -1 means using all processors.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - only most important warnings
                - 2 - shows all prints and all warnings.

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to an integer.
        """  # noqa
        self.model = model
        self.search_model = isinstance(model, BaseSearchCV)
        self.step = self._validate_step(step)
        self.min_features_to_select = self._validate_min_features(min_features_to_select)
        self.cv = cv
        self.scorer = get_single_scorer(scoring)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.report_df = pd.DataFrame()

    def compute(self):
        """
        Checks if fit() method has been run.

        and computes the DataFrame with results of feature elimination for each round.

        Returns:
            (pd.DataFrame):
                DataFrame with results of feature elimination for each round.
        """
        self._check_if_fitted()

        return self.report_df

    def fit_compute(
        self,
        X,
        y,
        sample_weight=None,
        columns_to_keep=None,
        column_names=None,
        shap_variance_penalty_factor=None,
        **shap_kwargs,
    ):
        """
        Fits the object with the provided data.

        The algorithm starts with the entire dataset, and then sequentially
            eliminates features. If sklearn compatible search CV is passed as model e.g.
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
                Labels for X.

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

            shap_variance_penalty_factor (int or float, optional):
                Apply aggregation penalty when computing average of shap values for a given feature.
                Results in a preference for features that have smaller standard deviation of shap
                values (more coherent shap importance). Recommend value 0.5 - 1.0.
                Formula: penalized_shap_mean = (mean_shap - (std_shap * shap_variance_penalty_factor))

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
            shap_variance_penalty_factor=shap_variance_penalty_factor,
            **shap_kwargs,
        )
        return self.compute()

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        columns_to_keep=None,
        column_names=None,
        groups=None,
        shap_variance_penalty_factor=None,
        **shap_kwargs,
    ):
        """
        Fits the object with the provided data.

        The algorithm starts with the entire dataset, and then sequentially
            eliminates features. If sklearn compatible search CV is passed as model e.g.
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
                Labels for X.

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

            groups (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,)
                Group labels for the samples used while splitting the dataset into train/test set.
                Only used in conjunction with a "Group" `cv` instance.
                (e.g. `sklearn.model_selection.GroupKFold`).

            shap_variance_penalty_factor (int or float, optional):
                Apply aggregation penalty when computing average of shap values for a given feature.
                Results in a preference for features that have smaller standard deviation of shap
                values (more coherent shap importance). Recommend value 0.5 - 1.0.
                Formula: penalized_shap_mean = (mean_shap - (std_shap * shap_variance_penalty_factor))

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.

        Returns:
            (ShapRFECV): Fitted object.
        """
        # Initialise len_columns_to_keep based on columns_to_keep content validation
        len_columns_to_keep = 0
        if columns_to_keep:
            if not all(isinstance(x, str) for x in columns_to_keep):
                raise ValueError("All elements in columns_to_keep must be strings.")
            len_columns_to_keep = len(columns_to_keep)

        # Validate matching column names, if both columns_to_keep and column_names are provided
        if column_names and not all(x in column_names for x in list(X.columns)):
            raise ValueError("Column names in columns_to_keep and column_names do not match.")

        # Validate total number of columns to select against the total number of columns
        if (
            column_names
            and columns_to_keep
            and (self.min_features_to_select + len_columns_to_keep) > len(self.column_names)
        ):
            raise ValueError("Minimum features to select plus columns_to_keep exceeds total number of features.")

        # Check shap_variance_penalty_factor has acceptable value
        if isinstance(shap_variance_penalty_factor, (float, int)) and shap_variance_penalty_factor >= 0:
            _shap_variance_penalty_factor = shap_variance_penalty_factor
        else:
            if shap_variance_penalty_factor is not None:
                warnings.warn(
                    "shap_variance_penalty_factor must be None, int or float. Setting shap_variance_penalty_factor = 0"
                )
            _shap_variance_penalty_factor = 0

        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, y_name="y", index=self.X.index, verbose=self.verbose)
        if sample_weight is not None:
            if self.verbose > 0:
                warnings.warn(
                    "sample_weight is passed only to the fit method of the model, not the evaluation metrics."
                )
            sample_weight = assure_pandas_series(sample_weight, index=self.X.index)
        self.cv = check_cv(self.cv, self.y, classifier=is_classifier(self.model))

        remaining_features = current_features_set = self.column_names
        round_number = 0

        # Stop when stopping criteria is met.
        stopping_criteria = np.max([self.min_features_to_select, len_columns_to_keep])

        # Setting up the min_features_to_select parameter.
        if columns_to_keep is not None:
            self.min_features_to_select = 0
            # Ensures that, if columns_to_keep is provided, the last features remaining are only the columns_to_keep.
            if self.verbose > 1:
                warnings.warn(f"Minimum features to select : {stopping_criteria}")

        while len(current_features_set) > stopping_criteria:
            round_number += 1

            # Get current dataset info
            current_features_set = remaining_features
            remaining_removeable_features = list(dict.fromkeys(current_features_set + (columns_to_keep or [])))

            # Current dataset
            current_X = self.X[remaining_removeable_features]

            # Optimize parameters
            if self.search_model:
                current_search_model = clone(self.model).fit(current_X, self.y)
                current_model = current_search_model.estimator.set_params(**current_search_model.best_params_)
            else:
                current_model = clone(self.model)

            # Perform CV to estimate feature importance with SHAP
            results_per_fold = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_feature_shap_values_per_fold)(
                    X=current_X,
                    y=self.y,
                    model=current_model,
                    train_index=train_index,
                    val_index=val_index,
                    sample_weight=sample_weight,
                    **shap_kwargs,
                )
                for train_index, val_index in self.cv.split(current_X, self.y, groups)
            )

            if self.y.nunique() == 2 or is_regressor(current_model):
                shap_values = np.concatenate([current_result[0] for current_result in results_per_fold], axis=0)
            else:  # multi-class case
                shap_values = np.concatenate([current_result[0] for current_result in results_per_fold], axis=1)

            scores_train = [current_result[1] for current_result in results_per_fold]
            scores_val = [current_result[2] for current_result in results_per_fold]

            # Calculate the shap features with remaining features and features to keep.
            shap_importance_df = calculate_shap_importance(
                shap_values, remaining_removeable_features, shap_variance_penalty_factor=_shap_variance_penalty_factor
            )

            # Determine which features to keep and which to remove.
            remaining_features, features_to_remove = self._filter_and_identify_features_based_on_importance(
                shap_importance_df, columns_to_keep, current_features_set
            )

            # Report results
            self._report_current_results(
                round_number=round_number,
                current_features_set=current_features_set,
                features_to_remove=features_to_remove,
                train_metric_mean=np.mean(scores_train),
                train_metric_std=np.std(scores_train),
                val_metric_mean=np.mean(scores_val),
                val_metric_std=np.std(scores_val),
            )
            if self.verbose > 1:
                logger.info(
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

    def plot(self, show=True, **figure_kwargs):
        """
        Generates plot of the model performance for each iteration of feature elimination.

        Args:
            show (bool, optional):
                If True, the plots are showed to the user, otherwise they are not shown. Not showing plot can be useful,
                when you want to edit the returned figure, before showing it.

            **figure_kwargs:
                Keyword arguments that are passed to the plt.figure, at its initialization.

        Returns:
            (plt.figure):
                Figure containing the performance plot.
        """
        # Data preparation
        num_features = self.report_df["num_features"]
        train_mean = self.report_df["train_metric_mean"]
        train_std = self.report_df["train_metric_std"]
        val_mean = self.report_df["val_metric_mean"]
        val_std = self.report_df["val_metric_std"]
        x_ticks = list(reversed(num_features.tolist()))

        # Plotting
        fig, ax = plt.subplots(**figure_kwargs)

        # Training performance
        ax.plot(num_features, train_mean, label="Train Score")
        ax.fill_between(num_features, train_mean - train_std, train_mean + train_std, alpha=0.3)

        # Validation performance
        ax.plot(num_features, val_mean, label="Validation Score")
        ax.fill_between(num_features, val_mean - val_std, val_mean + val_std, alpha=0.3)

        # Labels and title
        ax.set_xlabel("Number of features")
        ax.set_ylabel(f"Performance {self.scorer.metric_name}")
        ax.set_title("Backwards Feature Elimination using SHAP & CV")
        ax.legend(loc="lower left")
        ax.invert_xaxis()
        ax.set_xticks(x_ticks)

        # Display or close plot
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _validate_step(step):
        if not isinstance(step, (int, float)) or step <= 0:
            raise ValueError(f"Invalid step value: {step}. Must be a positive int or float.")
        return step

    @staticmethod
    def _validate_min_features(min_features):
        if not isinstance(min_features, int) or min_features <= 0:
            raise ValueError(f"Invalid min_features_to_select value: {min_features}. Must be a positive int.")
        return min_features

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
        # Calculate maximum nr of features that can be removed without dropping below
        # `min_num_features_to_keep`.
        nr_of_max_allowed_feature_removed = current_num_of_features - min_num_features_to_keep

        # Return smallest between `nr_of_max_allowed_feature_removed` and `num_features_to_remove`
        return min(num_features_to_remove, nr_of_max_allowed_feature_removed)

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

            columns_to_keep Optional(list)L
                A list of features that are kept.

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
            "features_set": [current_features_set],
            "eliminated_features": [features_to_remove],
            "train_metric_mean": train_metric_mean,
            "train_metric_std": train_metric_std,
            "val_metric_mean": val_metric_mean,
            "val_metric_std": val_metric_std,
        }

        if self.report_df.empty:
            self.report_df = pd.DataFrame(current_results, index=[round_number])
        else:
            new_row = pd.DataFrame(current_results, index=[round_number])
            # Append new_row to self.report_df more efficiently
            self.report_df = pd.concat([self.report_df, new_row])

    def _get_feature_shap_values_per_fold(
        self,
        X,
        y,
        model,
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
                Labels for X.

            model (classifier or regressor):
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
            model = model.fit(X_train, y_train, sample_weight=sample_weight.iloc[train_index])
        else:
            model = model.fit(X_train, y_train)

        # Score the model
        score_train = self.scorer.score(model, X_train, y_train)
        score_val = self.scorer.score(model, X_val, y_val)

        # Compute SHAP values
        shap_values = shap_calc(model, X_val, verbose=self.verbose, random_state=self.random_state, **shap_kwargs)
        return shap_values, score_train, score_val

    def _filter_and_identify_features_based_on_importance(
        self, shap_importance_df, columns_to_keep, current_features_set
    ):
        """
        Filters out features to be removed from the current feature set based on SHAP importance,
        while maintaining the original order of the features.

        Args:
            shap_importance_df (pd.DataFrame):
                A DataFrame containing the SHAP importance of the features.

            columns_to_keep (list):
                A list of column names that should not be removed, regardless of their
                SHAP importance.

            current_features_set (list):
                The current list of features from which features identified as
                less important will be removed. This list's order is maintained in the
                returned list of remaining features.

        Returns:
            remaining_features, features_to_remove (list, list): The features to keep & those that are removed.
        """
        # Get features to remove based on SHAP importance and columns to keep
        features_to_remove = self._get_current_features_to_remove(shap_importance_df, columns_to_keep=columns_to_keep)

        # Convert features_to_remove to a set for O(1) lookup times
        features_to_remove_set = set(features_to_remove)

        # Filter out the features to remove, maintaining the original order of current_features_set
        remaining_features = [feature for feature in current_features_set if feature not in features_to_remove_set]

        return remaining_features, features_to_remove

    def get_reduced_features_set(self, num_features, standard_error_threshold=1.0, return_type="feature_names"):
        """
        Gets the features set after the feature elimination process, for a given number of features.

        Args:
            num_features (int or str):
                If int: Number of features in the reduced features set.
                If str: One of the following automatic num feature selection methods supported:
                    1. best: strictly selects the num_features with the highest model score.
                    2. best_coherent: For iterations that are within standard_error_threshold of the highest
                    score, select the iteration with the lowest standard deviation of model score.
                    3. best_parsimonious: For iterations that are within standard_error_threshold of the
                    highest score, select the iteration with the fewest features.

            standard_error_threshold (float):
                If num_features is 'best_coherent' or 'best_parsimonious', this parameter is used.

            return_type:
                Accepts possible values of 'feature_names', 'support' or 'ranking'. These are defined as:
                    1. feature_names: returns column names
                    2. support: returns boolean mask
                    3. ranking: returns numeric ranking of features

        Returns:
            (list of str):
                Reduced features set.
        """
        self._check_if_fitted()

        # Determine the best number of features based on the method specified
        if isinstance(num_features, str):
            num_features = self._get_best_num_features(
                best_method=num_features, standard_error_threshold=standard_error_threshold
            )
        elif not isinstance(num_features, int):
            ValueError(
                "Parameter num_features can be of type int, or of type str with "
                "possible values of 'best', 'best_coherent' or 'best_parsimonious'"
            )

        # Get feature names for the determined number of features
        feature_names_selected = self._get_feature_names(num_features)

        # Return based on the requested return type
        if return_type == "feature_names":
            return feature_names_selected
        elif return_type == "support":
            return self._get_feature_support(feature_names_selected)
        elif return_type == "ranking":
            return self._get_feature_ranking()
        else:
            raise ValueError("Invalid return_type. Must be 'feature_names', 'support', or 'ranking'.")

    def _get_best_num_features(self, best_method, standard_error_threshold=1.0):
        """
        Helper function to identify the best number of features to select as per some automatic
        feature selection strategy. Strategies supported are:
            1. best: strictly selects the num_features with the highest model score.
            2. best_coherent: For iterations that are within standard_error_threshold of the highest
            score, select the iteration with the lowest standard deviation of model score.
            3. best_parsimonious: For iterations that are within standard_error_threshold of the
            highest score, select the iteration with the fewest features.

        Args:
            best_method (str):
                Automatic best feature selection strategy. One of "best", "best_coherent" or
                "best_parsimonious".

            standard_error_threshold (float):
                Parameter used if best_method is 'best_coherent' or 'best_parsimonious'.
                Numeric value greater than zero.

        Returns:
            (int)
                num_features as per automatic feature selection strategy selected.
        """
        self._check_if_fitted()

        if not isinstance(standard_error_threshold, (float, int)) or standard_error_threshold < 0:
            raise ValueError("Parameter standard_error_threshold must be a non-negative int or float.")

        # Perform copy after ValueError check.
        shap_report = self.report_df.copy()

        if best_method == "best":
            # Strictly selects the number of features with the highest model score
            best_score_index = shap_report["val_metric_mean"].idxmax()
            best_num_features = shap_report.loc[best_score_index, "num_features"]

        elif best_method == "best_coherent":
            # Selects within a threshold but prioritizes lower standard deviation
            highest_score = shap_report["val_metric_mean"].max()
            within_threshold = shap_report[shap_report["val_metric_mean"] >= highest_score - standard_error_threshold]
            lowest_std_index = within_threshold["val_metric_std"].idxmin()
            best_num_features = within_threshold.loc[lowest_std_index, "num_features"]

        elif best_method == "best_parsimonious":
            # Selects the fewest number of features within the threshold of the highest score
            highest_score = shap_report["val_metric_mean"].max()
            within_threshold = shap_report[shap_report["val_metric_mean"] >= highest_score - standard_error_threshold]
            fewest_features_index = within_threshold["num_features"].idxmin()
            best_num_features = within_threshold.loc[fewest_features_index, "num_features"]

        else:
            raise ValueError(
                "The parameter 'best_method' must be one of 'best', 'best_coherent', or 'best_parsimonious'."
            )

        # Log shap_report for users who want to inspect / debug
        if self.verbose > 1:
            logger.info(shap_report)

        return best_num_features

    def _get_feature_names(self, num_features):
        """
        Helper function that takes num_features and returns the associated list of column/feature names.

        Args:
            num_features (int):
                Represents the top N features to get the column names for.

        Returns:
            (list of feature names)
                List of the names of the features representing top num_features
        """
        self._check_if_fitted()

        # Direct lookup for the row with the desired number of features
        matching_rows = self.report_df[self.report_df.num_features == num_features]

        if matching_rows.empty:
            valid_nums = ", ".join([str(n) for n in sorted(self.report_df.num_features.unique())])
            raise ValueError(
                f"The provided number of features has not been achieved at any stage of the process. "
                f"You can select one of the following: {valid_nums}"
            )

        # Assuming 'features_set' contains the list of feature names for the row
        return matching_rows.iloc[0]["features_set"]

        # Assuming 'features_set' contains the list of feature names for the row
        return matching_rows.iloc[0]["features_set"]

    @staticmethod
    def _get_feature_support(self, feature_names_selected):
        """
        Helper function that takes feature_names_selected and returns a boolean mask representing the columns
        that were selected by the RFECV method.

        Args:
            feature_names_selected (list):
                Represents the top N features to get the column names for.

        Returns:
            (list of bools)
                Boolean mask representing the features selected.
        """
        support = [True if col in feature_names_selected else False for col in self.column_names]

        return support

    def _get_feature_ranking(self):
        """
        Returns the feature ranking, such that ranking_[i] corresponds to the ranking position
        of the i-th feature. Selected (i.e., estimated best) features are assigned rank 1.

        Returns:
            (list of bools)
                Boolean mask representing the features selected.
        """
        flipped_report_df = self.report_df.iloc[::-1]

        # Some features are not eliminated. All have importance of zero (highest importance)
        features_not_eliminated = flipped_report_df["features_set"].iloc[0]
        features_not_eliminated_dict = {v: 0 for v in features_not_eliminated}

        # Eliminated features are ranked by shap importance
        features_eliminated = np.concatenate(flipped_report_df["eliminated_features"].to_numpy())
        features_eliminated_dict = {int(v): k + 1 for (k, v) in enumerate(features_eliminated)}

        # Combine dicts with rank info
        features_eliminated_dict.update(features_not_eliminated_dict)

        # Get ranking per the order of columns
        ranking = [features_eliminated_dict[col] for col in self.column_names]

        return ranking
