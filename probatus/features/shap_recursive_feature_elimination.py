import warnings
from typing import Any, List, Optional, Tuple, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from probatus.core import BaseFitComputePlotClass
from probatus.features.shap_recursive_feature_elimination_helper import (
    check_if_model_is_compatible_with_early_stopping,
    get_feature_names,
    validate_shap_variance_penalty_factor_parameter,
    validate_step_parameter,
    validate_min_features_parameter,
    filter_and_identify_features_based_on_importance,
    report_current_results,
    get_best_num_features,
    get_feature_support,
    get_feature_ranking,
)
from probatus.features.shap_early_stopping_recursive_feature_elimination_helper import (
    get_fit_params,
)
from probatus.utils import (
    assure_pandas_series,
    calculate_shap_importance,
    preprocess_data,
    preprocess_labels,
    get_single_scorer,
    calculate_shap_explanation,
    Scorer,
    extract_shap_multiclass_params,
    shap_explanation_to_shap_df,
)
from probatus.utils.common import get_pipeline_preprocessor_and_estimator


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
            model (Union[BaseEstimator, BaseSearchCV, Pipeline]):
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
        if isinstance(model, Pipeline):
            self.pipeline, self.preprocessor = get_pipeline_preprocessor_and_estimator(model)
        else:
            self.pipeline = None
            self.preprocessor = None
        self.model = model
        self.search_model = isinstance(model, BaseSearchCV)
        self.step = validate_step_parameter(step)
        self.min_features_to_select = validate_min_features_parameter(min_features_to_select)
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

            if not check_if_model_is_compatible_with_early_stopping(model):
                raise ValueError("Only 'XGBoost', 'LGBM' and 'CatBoost' supported for early stopping.")

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        # Initialize attributes that will be set during fit
        self.report_df = pd.DataFrame()
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.column_names: Optional[List[str]] = None
        self.fitted = False

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

        # Transform data if model is a Pipeline
        if self.pipeline is not None:
            X = self.pipeline.transform(X)

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
        _shap_variance_penalty_factor = validate_shap_variance_penalty_factor_parameter(shap_variance_penalty_factor)

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
        with tqdm(total=max_iterations, desc="Feature Elimination") as progress_bar:
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
                remaining_features, features_to_remove = filter_and_identify_features_based_on_importance(
                    shap_importance_df, self.step, self.min_features_to_select, columns_to_keep, current_features_set
                )

                # Record results for this round
                self.report_df = report_current_results(
                    report_df=self.report_df,
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

    # TODO: Move logic to plot
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
        ax.grid(True, linestyle=":", alpha=0.7)

        # Show or close the plot based on the show parameter
        if show:
            plt.show()

        return fig

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
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values multi-classification conversion - parameters like 'class_selection', 'multiclass_aggregation', and 'weight_type'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multiclass models.

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

        # Split arguments for multi-classification
        multi_class_kwargs, shap_kwargs = extract_shap_multiclass_params(shap_kwargs)

        # Calculate SHAP values for validation set
        shap_explanation_val = calculate_shap_explanation(
            model, X_val, return_explainer=False, verbose=self.verbose, random_state=self.random_state, **shap_kwargs
        )

        shap_values_val = shap_explanation_to_shap_df(
            shap_explanation=shap_explanation_val,
            model=model,
            X=X_val,
            **multi_class_kwargs,
        )

        return shap_values_val, score_train, score_val

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
            num_features = get_best_num_features(
                report_df=self.report_df,
                best_method=num_features,
                standard_error_threshold=standard_error_threshold,
                verbose=self.verbose,
            )
        elif not isinstance(num_features, int):
            raise ValueError(
                "Parameter num_features must be an int or one of: 'best', 'best_coherent', 'best_parsimonious'"
            )

        # Get feature names for the determined number of features
        feature_names_selected = get_feature_names(report_df=self.report_df, num_features=num_features)

        # Return based on the requested return type
        if return_type == "feature_names":
            return feature_names_selected
        elif return_type == "support":
            return get_feature_support(column_names=self.column_names, feature_names_selected=feature_names_selected)
        elif return_type == "ranking":
            return get_feature_ranking(report_df=self.report_df)
        else:
            raise ValueError("Invalid return_type. Must be 'feature_names', 'support', or 'ranking'.")

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
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values multi-classification conversion - parameters like 'class_selection', 'multiclass_aggregation', and 'weight_type'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multiclass models.

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
        fit_params = get_fit_params(
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            train_index=train_index,
            val_index=val_index,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            verbose=self.verbose,
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

        # Split arguments for multi-classification
        multi_class_kwargs, shap_kwargs = extract_shap_multiclass_params(shap_kwargs)

        # Calculate SHAP values for validation set
        shap_explanation_val = calculate_shap_explanation(
            model, X_val, return_explainer=False, verbose=self.verbose, random_state=self.random_state, **shap_kwargs
        )

        shap_values_val = shap_explanation_to_shap_df(
            shap_explanation=shap_explanation_val,
            model=model,
            X=X_val,
            **multi_class_kwargs,
        )

        return shap_values_val, score_train, score_val
