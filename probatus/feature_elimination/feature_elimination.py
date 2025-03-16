import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV
from tqdm.auto import tqdm

from probatus.utils import (
    BaseFitComputePlotClass,
    assure_pandas_series,
    calculate_shap_importance,
    preprocess_data,
    preprocess_labels,
    get_single_scorer,
    shap_calc,
)
from probatus.utils import Scorer


class ShapRFECV(BaseFitComputePlotClass):
    """
    This class performs Backwards Recursive Feature Elimination, using SHAP feature importance.

    At each round, for a given feature set, starting from all available features, the following steps are applied:

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
        model: Union[BaseEstimator, BaseSearchCV],
        step: Union[int, float] = 1,
        min_features_to_select: int = 1,
        cv: Optional[Any] = None,
        scoring: Union[str, Scorer] = "roc_auc",
        n_jobs: int = -1,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
    ) -> None:
        """
        Initializes the ShapRFECV class for recursive feature elimination using SHAP importance.

        Args:
            model (Union[BaseEstimator, BaseSearchCV]):
                The model used for training and evaluation at each feature elimination step.
                - Recommended: `LGBMClassifier` (handles missing values and categorical features natively).
                - Supports hyperparameter tuning via `GridSearchCV`, `RandomizedSearchCV`, or `BayesSearchCV`.

            step (Union[int, float], optional):
                Number of features removed per iteration.
                - If `int`: Specifies the exact number of features removed in each round.
                - If `float`: Specifies the fraction of remaining features to remove (rounded down).
                Using a float is recommended for large feature sets, as it speeds up elimination in early stages
                and slows down for finer selection as fewer features remain.

            min_features_to_select (int, optional):
                The minimum number of features to retain before stopping the elimination process.
                - Default is `1`, meaning the process continues until only one feature is left.
                - If `columns_to_keep` is specified in the `fit` method, it may override this parameter.

            cv (Optional[Any], optional):
                Cross-validation strategy to use for model evaluation.
                - Compatible with sklearn's CV parameter formats.
                - If `None`, defaults to `cv=5`.

            scoring (Union[str, Scorer], default="roc_auc"):
                Metric for model performance evaluation. Can be either:
                - A string matching a predefined sklearn classification scorer name
                - A probatus.utils.Scorer instance for custom metrics

            n_jobs (int, optional):
                Number of CPU cores to use for parallel processing.
                - `None`: Uses a single core unless running in a `joblib.parallel_backend` context.
                - `-1`: Uses all available CPU cores.
                - Default is `-1`.

            verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
                - Default is `0`.

            random_state (Optional[int], optional):
                Controls the randomness of feature elimination and hyperparameter tuning.
                - If `None`, results will vary in different runs.
                - Set an integer value for reproducibility.
                - Default is `None`.

            early_stopping_rounds (Optional[int], optional):
                Number of rounds without performance improvement before stopping model training.
                - Only applies to SHAP value estimation (not hyperparameter tuning).
                - Only supported for `XGBoost`, `LightGBM`, and `CatBoost`.
                - Default is `None` (disabled).

            eval_metric (Optional[str], optional):
                Performance metric used to evaluate early stopping.
                - Only relevant if `early_stopping_rounds` is set.
                - Only supported by `XGBoost`, `LightGBM`, and `CatBoost`.
                - Default is `None`.

        Raises:
            ValueError:
                - If `early_stopping_rounds` is set without specifying `eval_metric`.
                - If `early_stopping_rounds` is not a positive integer.
                - If the model is not compatible with early stopping.

        """
        self.model = model
        self.search_model = isinstance(model, BaseSearchCV)
        self.step = self._validate_step(step)
        self.min_features_to_select = self._validate_min_features(min_features_to_select)
        self.cv = cv
        self.scorer = get_single_scorer(scoring)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        # Handle early stopping configuration
        if early_stopping_rounds:
            if not eval_metric:
                warnings.warn(
                    "Running early stopping requires both 'early_stopping_rounds' and 'eval_metric' as"
                    " parameters to be provided and supports only 'XGBoost', 'LGBM' and 'CatBoost'."
                )

            if not isinstance(early_stopping_rounds, int) or early_stopping_rounds <= 0:
                raise ValueError(f"early_stopping_rounds must be a positive integer; got {early_stopping_rounds}.")

            if not self._check_if_model_is_compatible_with_early_stopping(model):
                raise ValueError("Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping.")

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        # Initialize attributes that will be set during fit
        self.report_df = pd.DataFrame()
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.column_names: Optional[List[str]] = None
        self.fitted = False

    @staticmethod
    def _check_if_model_is_compatible_with_early_stopping(model: Union[BaseEstimator, BaseSearchCV]) -> bool:
        """
        Checks whether the given model supports early stopping.

        If the model is a hyperparameter search object (e.g., `GridSearchCV`, `RandomizedSearchCV`),
        this method checks its base estimator.

        Args:
            model (Union[BaseEstimator, BaseSearchCV]):
                The model or hyperparameter search object to check for early stopping compatibility.

        Returns:
            bool:
                `True` if the model supports early stopping, `False` otherwise.
        """
        # List of supported libraries and their model class names
        # TODO: Add CatBoostClassifier (when it supports NumPy 2.0)
        libraries = [("lightgbm", "LGBMModel"), ("xgboost", "XGBModel")]  # , ("catboost", "CatBoost")]

        # If model is a search CV, get the underlying estimator
        if isinstance(model, BaseSearchCV):
            model = model.estimator

        # Check if model is an instance of any supported class
        for lib, class_name in libraries:
            try:
                module = __import__(lib, fromlist=[class_name])
                model_class = getattr(module, class_name)
                if isinstance(model, model_class):
                    return True
            except (ImportError, AttributeError):
                # Skip if library is not installed or class doesn't exist
                warnings.warn(f"Library {lib} is not installed or class {class_name} does not exist.")

        return False

    def compute(self) -> pd.DataFrame:
        """
        Compute the DataFrame with results of feature elimination for each round.

        Returns:
            pd.DataFrame: DataFrame with results of feature elimination for each round.
        """
        self._check_if_fitted()
        return self.report_df

    def fit_compute(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        columns_to_keep: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        groups: Optional[pd.Series] = None,
        shap_variance_penalty_factor: Optional[Union[int, float]] = None,
        **shap_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fits the model and computes feature elimination results in a single step.

        This method performs SHAP-based recursive feature elimination while optionally
        optimizing hyperparameters at each iteration. The process follows these steps:

        1. Start with the full feature set.
        2. If a hyperparameter search object (e.g., `GridSearchCV`, `RandomizedSearchCV`, `BayesSearchCV`)
        is provided as the model, perform hyperparameter optimization at each step.
        3. Use cross-validation to compute SHAP feature importance.
        4. Remove the `step` lowest-importance features.
        5. Repeat until the minimum number of features is reached.
        6. Return a report containing results from each iteration.

        Args:
            X (pd.DataFrame):
                Input feature dataset of shape `(n_samples, n_features)`.

            y (pd.Series):
                Target labels of shape `(n_samples,)`.

            sample_weight (Optional[pd.Series], optional):
                Sample weights used for model fitting (if supported by the model).
                Note: Weights are applied only during training, not for metric calculation.
                Default is `None`.

            columns_to_keep (Optional[List[str]], optional):
                List of feature names that should not be eliminated.
                Default is `None`.

            column_names (Optional[List[str]], optional):
                Custom feature names to assign to `X`. If provided, this overwrites existing column names.
                Default is `None`.

            groups (Optional[pd.Series], optional):
                Group labels for samples when using `GroupKFold` cross-validation.
                Default is `None`.

            shap_variance_penalty_factor (Optional[Union[int, float]], optional):
                A penalty factor applied to SHAP values with high variance to reduce their influence.
                - Formula: `penalized_shap_mean = mean_shap - (std_shap * shap_variance_penalty_factor)`
                - Helps mitigate instability in SHAP-based rankings.
                - Recommended values: `0.5 - 1.0`.
                Default is `None` (no penalty applied).

            **shap_kwargs (Any):
                Additional keyword arguments passed to `shap.Explainer`.
                Common options:
                - `approximate` (bool): Enables faster but less accurate SHAP value computation.
                - `check_additivity` (bool): If `False`, disables SHAP additivity check.

        Returns:
            pd.DataFrame:
                A DataFrame containing feature elimination results from each iteration, including
                model performance metrics and selected feature sets.
        """
        self.fit(
            X,
            y,
            sample_weight=sample_weight,
            columns_to_keep=columns_to_keep,
            column_names=column_names,
            groups=groups,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
            **shap_kwargs,
        )
        return self.compute()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        columns_to_keep: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        groups: Optional[pd.Series] = None,
        shap_variance_penalty_factor: Optional[Union[int, float]] = None,
        **shap_kwargs: Any,
    ) -> "ShapRFECV":
        """
        Fits the model for SHAP-based recursive feature elimination.

        The process begins with the full feature set and iteratively removes the least important
        features based on SHAP values. If the model is wrapped in a hyperparameter search CV
        (e.g., `GridSearchCV`, `RandomizedSearchCV`, `BayesSearchCV`), tuning is performed at
        each elimination step.

        **Feature Elimination Steps:**
        1. Train the model (with optional hyperparameter tuning).
        2. Compute SHAP feature importance via cross-validation.
        3. Remove the `step` lowest-importance features.
        4. Repeat until the minimum feature count is reached.

        Args:
            X (pd.DataFrame):
                The input dataset of shape `(n_samples, n_features)`.

            y (pd.Series):
                Target labels corresponding to `X`.

            sample_weight (Optional[pd.Series], optional):
                Sample weights used for model training (if supported by the model).
                Note: Weights are applied only during model fitting, not during metric calculation.
                Default is `None`.

            columns_to_keep (Optional[List[str]], optional):
                List of feature names that will not be eliminated.
                These features are retained throughout the elimination process.
                Default is `None`.

            column_names (Optional[List[str]], optional):
                Custom feature names to assign to `X`. If provided, overwrites existing column names.
                Default is `None`.

            groups (Optional[pd.Series], optional):
                Group labels for samples when using `GroupKFold` cross-validation.
                Default is `None`.

            shap_variance_penalty_factor (Optional[Union[int, float]], optional):
                Penalization factor applied to SHAP values with high variance.
                - Formula: `penalized_shap_mean = mean_shap - (std_shap * shap_variance_penalty_factor)`
                - Helps reduce the influence of unstable SHAP values.
                - Recommended range: `0.5 - 1.0`.
                - Default is `None` (no penalty applied).

            **shap_kwargs (Any):
                Additional keyword arguments passed to `shap.Explainer`.
                Common options:
                - `approximate` (bool): Uses faster but less accurate SHAP calculation.
                - `check_additivity` (bool): If `False`, disables SHAP additivity check.

        Returns:
            ShapRFECV:
                The fitted instance with computed feature elimination results.

        Raises:
            ValueError:
                - If input data has an invalid format.
                - If `shap_variance_penalty_factor` is negative.
        """
        # Validate columns_to_keep
        len_columns_to_keep = 0
        if columns_to_keep:
            if not all(isinstance(x, str) for x in columns_to_keep):
                raise ValueError("All elements in columns_to_keep must be strings.")
            len_columns_to_keep = len(columns_to_keep)

        # Preprocess input data
        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, index=self.X.index)

        # Validate column names
        if column_names and not all(x in column_names for x in list(X.columns)):
            raise ValueError("Column names in columns_to_keep and column_names do not match.")

        # Validate total number of columns to select against the total number of columns
        if (
            column_names
            and columns_to_keep
            and (self.min_features_to_select + len_columns_to_keep) > len(self.column_names)
        ):
            raise ValueError("Minimum features to select plus columns_to_keep exceeds total number of features.")

        # Process sample weights if provided
        if sample_weight is not None:
            if self.verbose > 0:
                warnings.warn(
                    "sample_weight is passed only to the fit method of the model, not the evaluation metrics."
                )
            sample_weight = assure_pandas_series(sample_weight, index=self.X.index)

        # Validate and set shap_variance_penalty_factor
        if isinstance(shap_variance_penalty_factor, (float, int)) and shap_variance_penalty_factor >= 0:
            _shap_variance_penalty_factor = shap_variance_penalty_factor
        else:
            if shap_variance_penalty_factor is not None:
                warnings.warn(
                    "shap_variance_penalty_factor must be None, int or float. Setting shap_variance_penalty_factor = 0"
                )
            _shap_variance_penalty_factor = 0

        # Setup cross-validation
        self.cv = check_cv(self.cv, self.y, classifier=is_classifier(self.model))

        # Calculate stopping criteria
        stopping_criteria = max(self.min_features_to_select, len_columns_to_keep)

        # Adjust min_features_to_select if columns_to_keep is provided
        if columns_to_keep is not None:
            self.min_features_to_select = 0
            # Ensures that, if columns_to_keep is provided, the last features remaining are only the columns_to_keep.
            if self.verbose > 1:
                warnings.warn(f"Minimum features to select : {stopping_criteria}")

        # Initialize variables for the feature elimination loop
        remaining_features = current_features_set = self.column_names
        round_number = 0

        # Calculate the maximum number of iterations
        max_iterations = len(current_features_set) - stopping_criteria

        # Create a tqdm progress bar for feature elimination
        with tqdm(total=max_iterations, desc="Feature Elimination", disable=self.verbose == 0) as progress_bar:
            while len(current_features_set) > stopping_criteria:
                round_number += 1

                # Get current dataset info
                current_features_set = remaining_features
                remaining_removeable_features = list(dict.fromkeys(current_features_set + (columns_to_keep or [])))

                # Create current dataset with selected features
                current_X = self.X[remaining_removeable_features]

                # Optimize model parameters if using a search model
                if self.search_model:
                    current_search_model = clone(self.model).fit(current_X, self.y)
                    current_model = current_search_model.estimator.set_params(**current_search_model.best_params_)
                else:
                    current_model = clone(self.model)

                # Perform cross-validation to estimate feature importance with SHAP
                if not (self.early_stopping_rounds and self.eval_metric):
                    # Standard CV without early stopping
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
                else:
                    # CV with early stopping
                    results_per_fold = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._get_feature_shap_values_per_fold_early_stopping)(
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

                # Process SHAP values based on model type
                if self.y.nunique() == 2 or is_regressor(current_model):
                    # Binary classification or regression case
                    shap_values = np.concatenate([current_result[0] for current_result in results_per_fold], axis=0)
                else:
                    # Multi-class case
                    shap_values = np.concatenate([current_result[0] for current_result in results_per_fold], axis=1)

                # Extract scores from results
                scores_train = [current_result[1] for current_result in results_per_fold]
                scores_val = [current_result[2] for current_result in results_per_fold]

                # Calculate SHAP importance for features
                shap_importance_df = calculate_shap_importance(
                    shap_values,
                    remaining_removeable_features,
                    shap_variance_penalty_factor=_shap_variance_penalty_factor,
                )

                # Determine which features to keep and which to remove
                remaining_features, features_to_remove = self._filter_and_identify_features_based_on_importance(
                    shap_importance_df, columns_to_keep, current_features_set
                )

                # Record results for this round
                self._report_current_results(
                    round_number=round_number,
                    current_features_set=current_features_set,
                    features_to_remove=features_to_remove,
                    train_metric_mean=float(np.mean(scores_train)),
                    train_metric_std=float(np.std(scores_train)),
                    val_metric_mean=float(np.mean(scores_val)),
                    val_metric_std=float(np.std(scores_val)),
                )

                # Update the progress bar with the number of features removed in this iteration
                features_removed = len(current_features_set) - len(remaining_features)
                progress_bar.update(features_removed)

                # Update progress bar description with current performance
                progress_bar.set_description(
                    f"Feature Elimination (features: {len(remaining_features)}, val score: {self.report_df.loc[round_number]['val_metric_mean']:.4f})"
                )

                if self.verbose > 1:
                    logger.debug(
                        f"Round: {round_number}, Current number of features: {len(current_features_set)}, "
                        f"Current performance: Train {self.report_df.loc[round_number]['train_metric_mean']} "
                        f"+/- {self.report_df.loc[round_number]['train_metric_std']}, CV Validation "
                        f"{self.report_df.loc[round_number]['val_metric_mean']} "
                        f"+/- {self.report_df.loc[round_number]['val_metric_std']}. \n"
                        f"Features left: {remaining_features}. "
                        f"Removed features at the end of the round: {features_to_remove}"
                    )

        self.fitted = True
        return self

    def plot(self, show: bool = False, **figure_kwargs: Any) -> plt.Figure:
        """
        Plots model performance at each iteration of feature elimination.

        The visualization displays performance metrics (with error bars) for both
        training and validation sets as features are progressively removed.

        Args:
            show (bool, optional):
                Whether to display the plot immediately.
                - `True`: Displays the plot.
                - `False` (default): Returns the figure for further modifications before displaying.

            **figure_kwargs (Any):
                Additional keyword arguments passed to `plt.figure()` for customizing
                figure size, dpi, or other appearance settings.

        Returns:
            plt.Figure:
                The generated matplotlib figure object.

        Raises:
            RuntimeError:
                If called before the `fit` method is executed.
        """
        self._check_if_fitted()

        # Extract data from report
        num_features = self.report_df["num_features"]
        train_mean = self.report_df["train_metric_mean"]
        train_std = self.report_df["train_metric_std"]
        val_mean = self.report_df["val_metric_mean"]
        val_std = self.report_df["val_metric_std"]
        x_ticks = list(reversed(num_features.tolist()))

        # Create figure and axis
        fig, ax = plt.subplots(**figure_kwargs)

        # Plot training performance with error bars
        ax.errorbar(
            num_features, train_mean, yerr=train_std, fmt="o-", capsize=5, label="Train Score", markersize=8, alpha=0.7
        )

        # Plot validation performance with error bars
        ax.errorbar(
            num_features, val_mean, yerr=val_std, fmt="s-", capsize=5, label="Validation Score", markersize=8, alpha=0.7
        )

        # Configure plot appearance
        ax.set_xlabel("Number of features")
        ax.set_ylabel(f"Performance {self.scorer.metric_name}")
        ax.set_title("Backwards Feature Elimination using SHAP")
        ax.legend(loc="lower left")
        ax.invert_xaxis()  # Reverse x-axis to show feature reduction from left to right
        ax.set_xticks(x_ticks)
        # Rotate x-axis labels by 45 degrees
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Show or close the plot based on the show parameter
        if show:
            plt.show()

        return fig

    @staticmethod
    def _validate_step(step: Union[int, float]) -> Union[int, float]:
        """
        Validates the `step` parameter used for feature elimination.

        The step value determines how many features are removed per iteration.
        It must be a positive integer (absolute count) or a positive float
        (fraction of remaining features).

        Args:
            step (Union[int, float]):
                The step value to validate.

        Returns:
            Union[int, float]:
                The validated step value.

        Raises:
            ValueError:
                If `step` is not a positive integer or float.
        """
        if not isinstance(step, (int, float)) or step <= 0:
            raise ValueError(f"Invalid step value: {step}. Must be a positive int or float.")
        return step

    @staticmethod
    def _validate_min_features(min_features: int) -> int:
        """
        Validates the `min_features_to_select` parameter.

        Ensures that the minimum number of features to retain is a positive integer.

        Args:
            min_features (int):
                The minimum number of features to retain during feature elimination.

        Returns:
            int:
                The validated `min_features_to_select` value.

        Raises:
            ValueError:
                If `min_features` is not a positive integer.
        """
        if not isinstance(min_features, int) or min_features <= 0:
            raise ValueError(f"Invalid min_features_to_select value: {min_features}. Must be a positive int.")
        return min_features

    @staticmethod
    def _calculate_number_of_features_to_remove(
        current_num_of_features: int,
        num_features_to_remove: int,
        min_num_features_to_keep: int,
    ) -> int:
        """
        Calculates the number of features to remove while ensuring the minimum required features remain.

        This function ensures that feature elimination does not reduce the dataset below the
        specified `min_num_features_to_keep` threshold.

        Args:
            current_num_of_features (int):
                The current number of features in the dataset.

            num_features_to_remove (int):
                The proposed number of features to remove in this iteration.

            min_num_features_to_keep (int):
                The minimum number of features that must remain after feature removal.

        Returns:
            int:
                The adjusted number of features that can be safely removed without violating
                the `min_num_features_to_keep` constraint.
        """
        # Calculate maximum number of features that can be removed without dropping below min_num_features_to_keep
        max_allowed_features_to_remove = current_num_of_features - min_num_features_to_keep

        # Return the smaller of the two values to ensure we don't remove too many features
        return min(num_features_to_remove, max_allowed_features_to_remove)

    def _get_current_features_to_remove(
        self, shap_importance_df: pd.DataFrame, columns_to_keep: Optional[List[str]] = None
    ) -> List[str]:
        """
        Determines which features to remove based on SHAP importance.

        This method selects features for removal according to the `step` parameter:
        - If `step` is an integer: Removes exactly that many lowest-importance features (if available).
        - If `step` is a float: Removes that fraction of the remaining features (rounded down).

        Args:
            shap_importance_df (pd.DataFrame):
                DataFrame containing SHAP importance values for features.

            columns_to_keep (Optional[List[str]], optional):
                List of feature names that should not be removed. These features are
                excluded from consideration when selecting features for removal.
                Default is `None`.

        Returns:
            List[str]:
                A list of feature names selected for removal in the current iteration.
        """
        # Bounding the variable.
        num_features_to_remove = 0

        # Exclude columns_to_keep from consideration for removal
        if columns_to_keep is not None:
            mask = shap_importance_df.index.isin(columns_to_keep)
            shap_importance_df = shap_importance_df[~mask]

        # Calculate number of features to remove based on step type
        if isinstance(self.step, int):
            num_features_to_remove = self._calculate_number_of_features_to_remove(
                current_num_of_features=shap_importance_df.shape[0],
                num_features_to_remove=self.step,
                min_num_features_to_keep=self.min_features_to_select,
            )
        # If the step is a float remove n * number features that are left, rounded down
        elif isinstance(self.step, float):
            current_step = int(np.floor(shap_importance_df.shape[0] * self.step))

            # Ensure at least 1 feature is removed (if possible)
            if current_step < 1:
                current_step = 1

            num_features_to_remove = self._calculate_number_of_features_to_remove(
                current_num_of_features=shap_importance_df.shape[0],
                num_features_to_remove=current_step,
                min_num_features_to_keep=self.min_features_to_select,
            )

        # Return empty list if no features should be removed
        if num_features_to_remove == 0:
            return []

        # Return the n features with lowest importance
        return shap_importance_df.iloc[-num_features_to_remove:].index.tolist()

    def _report_current_results(
        self,
        round_number: int,
        current_features_set: List[str],
        features_to_remove: List[str],
        train_metric_mean: float,
        train_metric_std: float,
        val_metric_mean: float,
        val_metric_std: float,
    ) -> None:
        """
        Records the results of the current feature elimination iteration.

        This method updates the report DataFrame with details about the current
        feature set, removed features, and model performance metrics.

        Args:
            round_number (int):
                The current iteration number of feature elimination.

            current_features_set (List[str]):
                The list of features used in this iteration before feature removal.

            features_to_remove (List[str]):
                The list of features scheduled for removal after this iteration.

            train_metric_mean (float):
                The mean performance metric for the training set.

            train_metric_std (float):
                The standard deviation of the training set performance metric.

            val_metric_mean (float):
                The mean performance metric for the validation set.

            val_metric_std (float):
                The standard deviation of the validation set performance metric.
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

        # Create a new DataFrame row with the current results
        new_row = pd.DataFrame(current_results, index=[round_number])

        # Add the new row to the report DataFrame
        if self.report_df.empty:
            self.report_df = new_row
        else:
            self.report_df = pd.concat([self.report_df, new_row])

    def _get_feature_shap_values_per_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Union[BaseEstimator, BaseSearchCV],
        train_index: np.ndarray,
        val_index: np.ndarray,
        sample_weight: Optional[pd.Series] = None,
        **shap_kwargs: Any,
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Computes SHAP values and model scores for a single cross-validation fold.

        This method fits the model on the training subset and evaluates it on the
        validation subset, extracting SHAP values for feature importance analysis.

        Args:
            X (pd.DataFrame):
                The input dataset of shape `(n_samples, n_features)`.

            y (pd.Series):
                Target labels corresponding to `X`.

            model (Union[BaseEstimator, BaseSearchCV]):
                The model or hyperparameter search object used for training.

            train_index (np.ndarray):
                Indices of training samples for this cross-validation fold.

            val_index (np.ndarray):
                Indices of validation samples for this cross-validation fold.

            sample_weight (Optional[pd.Series], optional):
                Optional weights for samples during model training. Only used if
                the model supports sample weighting.
                Default is `None`.

            **shap_kwargs (Any):
                Additional keyword arguments passed to SHAP explainer.

        Returns:
            Tuple[pd.DataFrame, float, float]:
                A tuple containing:
                - `pd.DataFrame`: SHAP values for validation samples.
                - `float`: Training score for this fold.
                - `float`: Validation score for this fold.
        """
        # Split data into train and validation sets for this fold
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Fit the model with or without sample weights
        if sample_weight is not None:
            model = model.fit(X_train, y_train, sample_weight=sample_weight.iloc[train_index])
        else:
            model = model.fit(X_train, y_train)

        # Calculate performance scores
        score_train = self.scorer.score(model, X_train, y_train)
        score_val = self.scorer.score(model, X_val, y_val)

        # Calculate SHAP values for validation set
        shap_values = shap_calc(
            model, X_val, return_explainer=False, verbose=self.verbose, random_state=self.random_state, **shap_kwargs
        )

        return shap_values, score_train, score_val

    def _filter_and_identify_features_based_on_importance(
        self, shap_importance_df: pd.DataFrame, columns_to_keep: Optional[List[str]], current_features_set: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Filters features based on SHAP importance while preserving the original feature order.

        This method determines which features should be kept and which should be removed
        in the current iteration of feature elimination.

        Args:
            shap_importance_df (pd.DataFrame):
                DataFrame containing SHAP importance values for features.

            columns_to_keep (Optional[List[str]]):
                List of feature names that should not be removed, ensuring they remain in the dataset.
                Default is `None`.

            current_features_set (List[str]):
                List of currently selected features, preserving the original feature order.

        Returns:
            Tuple[List[str], List[str]]:
                A tuple containing:
                - `List[str]`: Features to keep (`remaining_features`).
                - `List[str]`: Features selected for removal (`features_to_remove`).
        """
        # Get features to remove based on SHAP importance and columns_to_keep
        features_to_remove = self._get_current_features_to_remove(shap_importance_df, columns_to_keep=columns_to_keep)

        # Convert features_to_remove to a set for O(1) lookup times
        features_to_remove_set = set(features_to_remove)

        # Filter out the features to remove, maintaining the original order of current_features_set
        remaining_features = [feature for feature in current_features_set if feature not in features_to_remove_set]

        return remaining_features, features_to_remove

    def get_reduced_features_set(
        self,
        num_features: Union[int, Literal["best", "best_coherent", "best_parsimonious"]],
        standard_error_threshold: float = 1.0,
        return_type: Literal["feature_names", "support", "ranking"] = "feature_names",
    ) -> Union[List[str], List[bool], List[int]]:
        """
        Retrieves the optimal set of selected features after feature elimination.

        Feature selection can be based on a fixed number of features or an automatic
        strategy that considers validation performance and stability.

        Args:
            num_features (Union[int, Literal["best", "best_coherent", "best_parsimonious"]]):
                Specifies how many features to select:
                - If `int`: Selects exactly that many features.
                - If `str`: Uses one of the following automatic selection strategies:
                    - `"best"`: Chooses features from the iteration with the highest validation score.
                    - `"best_coherent"`: Among iterations within `standard_error_threshold` of the best score,
                    selects the iteration with the lowest standard deviation.
                    - `"best_parsimonious"`: Among iterations within `standard_error_threshold` of the best score,
                    selects the iteration with the fewest features.

            standard_error_threshold (float, optional):
                The threshold for considering an iteration as sufficiently close to the best score.
                Used only when `num_features` is `"best_coherent"` or `"best_parsimonious"`.
                Default is `1.0`.

            return_type (Literal["feature_names", "support", "ranking"], optional):
                Specifies the format of the returned feature selection result:
                - `"feature_names"` (default): Returns a list of selected feature names.
                - `"support"`: Returns a boolean mask where `True` indicates selected features.
                - `"ranking"`: Returns a numeric ranking where lower values indicate more important features.

        Returns:
            Union[List[str], List[bool], List[int]]:
                The selected features in the format specified by `return_type`.

        Raises:
            ValueError:
                - If `num_features` is not a valid integer or one of `"best"`, `"best_coherent"`, or `"best_parsimonious"`.
                - If `return_type` is not one of `"feature_names"`, `"support"`, or `"ranking"`.
        """
        self._check_if_fitted()

        # Determine the best number of features based on the method specified
        if isinstance(num_features, str):
            num_features = self._get_best_num_features(
                best_method=num_features, standard_error_threshold=standard_error_threshold
            )
        elif not isinstance(num_features, int):
            raise ValueError(
                "Parameter num_features must be an int or one of: 'best', 'best_coherent', 'best_parsimonious'"
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

    def _get_best_num_features(
        self, best_method: Literal["best", "best_coherent", "best_parsimonious"], standard_error_threshold: float = 1.0
    ) -> int:
        """
        Determines the optimal number of features based on the specified selection strategy.

        Args:
            best_method (Literal["best", "best_coherent", "best_parsimonious"]):
                The strategy used to select the optimal number of features:
                - `"best"`: Chooses the iteration with the highest validation score.
                - `"best_coherent"`: Among iterations within `standard_error_threshold` of the best score,
                selects the one with the lowest standard deviation.
                - `"best_parsimonious"`: Among iterations within `standard_error_threshold` of the best score,
                selects the iteration with the fewest features.

            standard_error_threshold (float, optional):
                Defines how close an iteration's score must be to the highest score for it to be considered
                when using `"best_coherent"` or `"best_parsimonious"`.
                Default is `1.0`.

        Returns:
            int:
                The optimal number of features based on the selected strategy.

        Raises:
            ValueError:
                - If `best_method` is not one of `"best"`, `"best_coherent"`, or `"best_parsimonious"`.
                - If `standard_error_threshold` is negative or otherwise invalid.
        """
        self._check_if_fitted()

        if not isinstance(standard_error_threshold, (float, int)) or standard_error_threshold < 0:
            raise ValueError("Parameter standard_error_threshold must be a non-negative int or float.")

        # Create a copy of the report DataFrame to avoid modifying the original
        shap_report = self.report_df.copy()

        if best_method == "best":
            # Simply select the iteration with the highest validation score
            best_score_index = shap_report["val_metric_mean"].idxmax()
            best_num_features = shap_report.loc[best_score_index, "num_features"]

        elif best_method == "best_coherent":
            # Find the highest validation score
            highest_score = shap_report["val_metric_mean"].max()

            # Select iterations within the threshold of the highest score
            within_threshold = shap_report[shap_report["val_metric_mean"] >= highest_score - standard_error_threshold]

            # From those iterations, select the one with the lowest standard deviation
            lowest_std_index = within_threshold["val_metric_std"].idxmin()
            best_num_features = within_threshold.loc[lowest_std_index, "num_features"]

        elif best_method == "best_parsimonious":
            # Find the highest validation score
            highest_score = shap_report["val_metric_mean"].max()

            # Select iterations within the threshold of the highest score
            within_threshold = shap_report[shap_report["val_metric_mean"] >= highest_score - standard_error_threshold]

            # From those iterations, select the one with the fewest features
            fewest_features_index = within_threshold["num_features"].idxmin()
            best_num_features = within_threshold.loc[fewest_features_index, "num_features"]

        else:
            raise ValueError(
                "The parameter 'best_method' must be one of 'best', 'best_coherent', or 'best_parsimonious'."
            )

        # Log the report if verbose
        if self.verbose > 1:
            logger.debug(shap_report)

        return best_num_features

    def _get_feature_names(self, num_features: int) -> List[str]:
        """
        Retrieves the list of feature names corresponding to a specific number of selected features.

        Args:
            num_features (int):
                The number of features to retrieve.

        Returns:
            List[str]:
                A list of feature names corresponding to the specified number of features.

        Raises:
            ValueError:
                If the requested `num_features` was not achieved during the feature elimination process.
        """
        self._check_if_fitted()

        # Find the row in the report with the specified number of features
        matching_rows = self.report_df[self.report_df.num_features == num_features]

        if matching_rows.empty:
            valid_nums = ", ".join([str(n) for n in sorted(self.report_df.num_features.unique())])
            raise ValueError(
                f"The provided number of features ({num_features}) was not achieved during feature elimination. "
                f"Valid options are: {valid_nums}"
            )

        # Return the feature set from the first matching row
        return matching_rows.iloc[0]["features_set"]

    def _get_feature_support(self, feature_names_selected: List[str]) -> List[bool]:
        """
        Generates a boolean mask indicating which features were selected.

        Args:
            feature_names_selected (List[str]):
                A list of feature names that were selected after feature elimination.

        Returns:
            List[bool]:
                A boolean mask where `True` indicates a selected feature and `False` indicates a removed feature.
        """
        # Create a boolean mask where True indicates the feature was selected
        if self.column_names is None:
            raise ValueError("Feature names are not available. The model may not be fitted.")
        return [col in feature_names_selected for col in self.column_names]

    def _get_feature_ranking(self) -> List[int]:
        """
        Retrieves the ranking of features based on their elimination order.

        Features eliminated earlier receive higher ranks (indicating lower importance),
        while features that were never eliminated receive a rank of `0` (most important).

        Returns:
            List[int]:
                A list of feature rankings, where lower values indicate more important features
                and higher values correspond to features eliminated earlier.
        """
        # Reverse the report DataFrame to process elimination in chronological order
        flipped_report_df = self.report_df.iloc[::-1]

        # Features that were never eliminated have the highest importance (rank 0)
        features_not_eliminated = flipped_report_df["features_set"].iloc[0]
        features_not_eliminated_dict = {feature: 0 for feature in features_not_eliminated}

        # Features that were eliminated are ranked by their elimination order
        # Earlier elimination = higher rank number = less important
        features_eliminated = np.concatenate(flipped_report_df["eliminated_features"].to_numpy())
        features_eliminated_dict = {feature: idx + 1 for idx, feature in enumerate(features_eliminated)}

        # Combine the dictionaries
        ranking_dict = {**features_eliminated_dict, **features_not_eliminated_dict}

        # Create the ranking list in the original column order
        if self.column_names is None:
            raise ValueError("Feature names are not available. The model may not be fitted.")
        ranking = [ranking_dict.get(col, 0) for col in self.column_names]

        return ranking

    def _get_fit_params_lightGBM(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        train_index: Optional[np.ndarray] = None,
        val_index: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Prepares the fit parameters for LightGBM models with early stopping support.

        This method formats the necessary parameters for training LightGBM, including
        validation data and optional sample weights.

        Args:
            X_train (pd.DataFrame):
                Training feature matrix of shape `(n_samples, n_features)`.

            y_train (pd.Series):
                Training labels of shape `(n_samples,)`.

            X_val (pd.DataFrame):
                Validation feature matrix of shape `(n_samples, n_features)`.

            y_val (pd.Series):
                Validation labels of shape `(n_samples,)`.

            sample_weight (Optional[pd.Series], optional):
                Sample weights for training data, if applicable. Default is `None`.

            train_index (Optional[np.ndarray], optional):
                Indices of training samples. Default is `None`.

            val_index (Optional[np.ndarray], optional):
                Indices of validation samples. Default is `None`.

        Returns:
            Dict[str, Any]:
                A dictionary containing the formatted parameters to be passed to
                the LightGBM `fit` method, including validation sets and early stopping criteria.
        """
        from lightgbm import early_stopping, log_evaluation

        # Ensure early_stopping_rounds is not None
        if self.early_stopping_rounds is None:
            raise ValueError("early_stopping_rounds must be provided for LightGBM early stopping")

        # Create the fit parameters with eval_set and callbacks
        fit_params = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": self.eval_metric,
            "callbacks": [
                early_stopping(self.early_stopping_rounds, first_metric_only=True),
                log_evaluation(1 if self.verbose > 1 else 0),
            ],
        }

        # Add sample weights if provided
        if sample_weight is not None and train_index is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_index]

            # Add validation sample weights if validation indices are provided
            if val_index is not None:
                fit_params["eval_sample_weight"] = [sample_weight.iloc[val_index]]

        return fit_params

    def _get_fit_params_XGBoost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        train_index: Optional[np.ndarray] = None,
        val_index: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Prepares the fit parameters for training an XGBoost model with early stopping support.

        This method formats the required parameters for XGBoost training, ensuring the
        correct structure for validation data and sample weights. The early stopping
        configuration is handled separately via the model's parameters.

        Args:
            X_train (pd.DataFrame):
                Training feature matrix of shape `(n_samples, n_features)`.

            y_train (pd.Series):
                Training labels of shape `(n_samples,)`.

            X_val (pd.DataFrame):
                Validation feature matrix of shape `(n_samples, n_features)`.

            y_val (pd.Series):
                Validation labels of shape `(n_samples,)`.

            sample_weight (Optional[pd.Series], optional):
                Sample weights for training data, if applicable.
                Default is `None`.

            train_index (Optional[np.ndarray], optional):
                Indices of training samples.
                Default is `None`.

            val_index (Optional[np.ndarray], optional):
                Indices of validation samples.
                Default is `None`.

        Returns:
            Dict[str, Any]:
                A dictionary containing the formatted parameters to be passed to the
                XGBoost `fit` method, including validation data.
        """
        # Create fit parameters dictionary with eval_set for validation
        fit_params = {"eval_set": [(X_val, y_val)]}

        # Add sample weights if provided
        if sample_weight is not None and train_index is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_index]

            # Add validation sample weights if validation indices are provided
            if val_index is not None:
                fit_params["sample_weight_eval_set"] = [sample_weight.iloc[val_index]]

        return fit_params

    def _get_fit_params_CatBoost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        train_index: Optional[np.ndarray] = None,
        val_index: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Prepares the fit parameters for training a CatBoost model with early stopping support.

        This method structures the necessary parameters for CatBoost training, ensuring the correct
        format for validation data, optional sample weights, and early stopping.

        Args:
            X_train (pd.DataFrame):
                Training feature matrix of shape `(n_samples, n_features)`.

            y_train (pd.Series):
                Training labels of shape `(n_samples,)`.

            X_val (pd.DataFrame):
                Validation feature matrix of shape `(n_samples, n_features)`.

            y_val (pd.Series):
                Validation labels of shape `(n_samples,)`.

            sample_weight (Optional[pd.Series], optional):
                Sample weights for training data, if applicable.
                Default is `None`.

            train_index (Optional[np.ndarray], optional):
                Indices of training samples.
                Default is `None`.

            val_index (Optional[np.ndarray], optional):
                Indices of validation samples.
                Default is `None`.

        Returns:
            Dict[str, Any]:
                A dictionary containing the formatted parameters to be passed to the
                CatBoost `fit` method, including validation data and early stopping settings.
        """
        from catboost import Pool

        # Identify categorical features
        cat_features = [col for col in X_train.select_dtypes(include=["category"]).columns]

        # Create validation data pool
        eval_set_pool = Pool(X_val, y_val, cat_features=cat_features)

        # Add validation sample weights if provided
        if sample_weight is not None and val_index is not None:
            eval_set_pool.set_weight(sample_weight.iloc[val_index])

        # Create fit parameters dictionary with eval_set
        fit_params = {"eval_set": eval_set_pool, "cat_features": cat_features}

        # Add sample weights if provided
        if sample_weight is not None and train_index is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_index]

        return fit_params

    def _get_fit_params(
        self,
        model: Union[BaseEstimator, BaseSearchCV],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        train_index: Optional[np.ndarray] = None,
        val_index: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generates the appropriate fit parameters based on the model type.

        This method automatically detects the model type and delegates to the corresponding
        specialized method (`_get_fit_params_XGBoost`, `_get_fit_params_CatBoost`,
        `_get_fit_params_lightGBM`) to retrieve the correct fit parameters.

        Args:
            model (Union[BaseEstimator, BaseSearchCV]):
                The model or hyperparameter search object for which to retrieve fit parameters.

            X_train (pd.DataFrame):
                Training feature matrix of shape `(n_samples, n_features)`.

            y_train (pd.Series):
                Training labels of shape `(n_samples,)`.

            X_val (pd.DataFrame):
                Validation feature matrix of shape `(n_samples, n_features)`.

            y_val (pd.Series):
                Validation labels of shape `(n_samples,)`.

            sample_weight (Optional[pd.Series], optional):
                Sample weights for training data, if applicable.
                Default is `None`.

            train_index (Optional[np.ndarray], optional):
                Indices of training samples.
                Default is `None`.

            val_index (Optional[np.ndarray], optional):
                Indices of validation samples.
                Default is `None`.

        Returns:
            Dict[str, Any]:
                A dictionary of parameters to be passed to the model's `fit` method,
                including validation data and early stopping settings.

        Raises:
            ValueError:
                If the model type is not supported for early stopping.
        """
        # Try LightGBM
        try:
            from lightgbm import LGBMModel

            if isinstance(model, LGBMModel):
                return self._get_fit_params_lightGBM(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    sample_weight=sample_weight,
                    train_index=train_index,
                    val_index=val_index,
                )
        except ImportError:
            pass

        # Try XGBoost
        try:
            from xgboost import XGBModel

            if isinstance(model, XGBModel):
                return self._get_fit_params_XGBoost(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    sample_weight=sample_weight,
                    train_index=train_index,
                    val_index=val_index,
                )
        except ImportError:
            pass

        # TODO: Revert this once CatBoost is updated to work with NumPy 2.0
        try:
            # Only attempt to import if the model's class name suggests it might be a CatBoost model
            if hasattr(model, "__class__") and "catboost" in str(model.__class__).lower():
                try:
                    from catboost import CatBoost

                    if isinstance(model, CatBoost):
                        return self._get_fit_params_CatBoost(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            sample_weight=sample_weight,
                            train_index=train_index,
                            val_index=val_index,
                        )
                except ImportError:
                    pass
        except Exception:
            # Ignore any errors during this check
            pass

        raise ValueError("Model type not supported for early stopping")

    def _get_feature_shap_values_per_fold_early_stopping(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Union[BaseEstimator, BaseSearchCV],
        train_index: np.ndarray,
        val_index: np.ndarray,
        sample_weight: Optional[pd.Series] = None,
        **shap_kwargs: Any,
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Computes SHAP values and model scores for a cross-validation fold with early stopping.

        This method extends `_get_feature_shap_values_per_fold` by incorporating early stopping
        for models that support it, improving training efficiency.

        Args:
            X (pd.DataFrame):
                Feature dataset of shape `(n_samples, n_features)`.

            y (pd.Series):
                Target labels of shape `(n_samples,)`.

            model (Union[BaseEstimator, BaseSearchCV]):
                The model or hyperparameter search object to be trained.

            train_index (np.ndarray):
                Indices of training samples for this cross-validation fold.

            val_index (np.ndarray):
                Indices of validation samples for this cross-validation fold.

            sample_weight (Optional[pd.Series], optional):
                Sample weights for training data, if applicable.
                Default is `None`.

            **shap_kwargs (Any):
                Additional keyword arguments passed to `shap.Explainer`.

        Returns:
            Tuple[np.ndarray, float, float]:
                A tuple containing:
                - `pd.DataFrame`: SHAP values for validation samples.
                - `float`: Training score for this fold.
                - `float`: Validation score for this fold.
        """
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Get appropriate fit parameters for the model type
        fit_params = self._get_fit_params(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            train_index=train_index,
            val_index=val_index,
        )

        try:
            from xgboost import XGBModel

            if isinstance(model, XGBModel):
                model.set_params(eval_metric=self.eval_metric, early_stopping_rounds=self.early_stopping_rounds)
        except ImportError:
            pass

        # TODO: Revert this once CatBoost is updated to work with NumPy 2.0
        try:
            # Only attempt to import if the model's class name suggests it might be a CatBoost model
            if hasattr(model, "__class__") and "catboost" in str(model.__class__).lower():
                try:
                    from catboost import CatBoost

                    if isinstance(model, CatBoost):
                        model.set_params(early_stopping_rounds=self.early_stopping_rounds)
                except ImportError:
                    pass
        except Exception:
            # Ignore any errors during this check
            pass

        # Train the model with early stopping
        # For XGBoost and LightGBM, we need to pass X_train and y_train explicitly
        # since they're no longer included in fit_params
        model = model.fit(X_train, y_train, **fit_params)

        # Calculate performance scores
        score_train = self.scorer.score(model, X_train, y_train)
        score_val = self.scorer.score(model, X_val, y_val)

        # Calculate SHAP values for validation set
        shap_values = shap_calc(model, X_val, verbose=self.verbose, random_state=self.random_state, **shap_kwargs)

        return shap_values, score_train, score_val
