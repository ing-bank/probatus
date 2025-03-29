from typing import Union, Optional, List, Tuple, Literal
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from warnings import warn


def check_if_model_is_compatible_with_early_stopping(model: Union[BaseEstimator, BaseSearchCV]) -> bool:
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
            warn(f"Library {lib} is not installed or class {class_name} does not exist.")

    return False


def validate_step_parameter(step: Union[int, float]) -> Union[int, float]:
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
    if not isinstance(step, (int, np.int64, float, np.float64)):
        raise TypeError(f"step must be an integer; got {type(step)}.")
    if step <= 0:
        raise ValueError(f"min_features_to_select must be > 0; got {step}.")
    return step


def validate_min_features_parameter(min_features: int) -> int:
    """
    Validates the min_features_to_select parameter.

    Args:
        min_features (int):
            The minimum number of features to select.

    Returns:
        int:
            The validated min_features value.

    Raises:
        ValueError:
            If min_features is not a positive integer.
    """
    if not isinstance(min_features, (int, np.int64)):
        raise TypeError(f"min_features_to_select must be an integer; got {type(min_features)}.")
    if min_features <= 0:
        raise ValueError(f"min_features_to_select must be > 0; got {min_features}.")
    return min_features


def validate_shap_variance_penalty_factor_parameter(shap_variance_penalty_factor: Optional[Union[int, float]]) -> float:
    """
    Validates the shap_variance_penalty_factor parameter.

    Args:
        shap_variance_penalty_factor (Optional[Union[int, float]]):
            The penalty factor to apply to SHAP values with high variance.

    Returns:
        float:
            The validated shap_variance_penalty_factor value.
    """
    if (
        isinstance(shap_variance_penalty_factor, (int, np.int64, float, np.float64))
        and shap_variance_penalty_factor >= 0
    ):
        return float(shap_variance_penalty_factor)
    else:
        if shap_variance_penalty_factor is not None:
            warn("shap_variance_penalty_factor must be None, int or float. Setting shap_variance_penalty_factor = 0")
        return 0.0


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
    shap_importance_df: pd.DataFrame,
    step: Union[int, float],
    min_features_to_select: int,
    columns_to_keep: Optional[List[str]] = None,
) -> List[str]:
    """
    Determines which features to remove based on SHAP importance.

    This method selects features for removal according to the `step` parameter:
    - If `step` is an integer: Removes exactly that many lowest-importance features (if available).
    - If `step` is a float: Removes that fraction of the remaining features (rounded down).

    Output order is from worst to best features.

    Args:
        shap_importance_df (pd.DataFrame):
            DataFrame containing SHAP importance values for features.

        step (Union[int, float]):
            The number of features to remove in each iteration.

        min_features_to_select (int):
            The minimum number of features to keep in the dataset.

        columns_to_keep (Optional[List[str]], optional):
            List of feature names that should not be removed. These features are
            excluded from consideration when selecting features for removal.
            Default is `None`.

    Returns:
        List[str]:
            A list of feature names selected for removal in the current iteration.
            The order is from worst to best features.
    """
    # Bounding the variable.
    num_features_to_remove = 0

    # Exclude columns_to_keep from consideration for removal
    if columns_to_keep is not None:
        mask = shap_importance_df.index.isin(columns_to_keep)
        shap_importance_df = shap_importance_df[~mask]

    # Calculate number of features to remove based on step type
    if isinstance(step, (int, np.int64)):
        num_features_to_remove = _calculate_number_of_features_to_remove(
            current_num_of_features=shap_importance_df.shape[0],
            num_features_to_remove=int(step),
            min_num_features_to_keep=min_features_to_select,
        )
    # If the step is a float remove n * number features that are left, rounded down
    elif isinstance(step, (float, np.float64)):
        current_step = int(np.floor(shap_importance_df.shape[0] * step))

        # Ensure at least 1 feature is removed (if possible)
        if current_step < 1:
            current_step = 1

        num_features_to_remove = _calculate_number_of_features_to_remove(
            current_num_of_features=shap_importance_df.shape[0],
            num_features_to_remove=current_step,
            min_num_features_to_keep=min_features_to_select,
        )

    # Return empty list if no features should be removed
    if num_features_to_remove == 0:
        return []

    # Return the n features with lowest importance
    # Order: worst features first, best features last
    return shap_importance_df.iloc[-num_features_to_remove:].index.tolist()[::-1]


def filter_and_identify_features_based_on_importance(
    shap_importance_df: pd.DataFrame,
    step: Union[int, float],
    min_features_to_select: int,
    columns_to_keep: Optional[List[str]],
    current_features_set: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Determines which features should be kept and which should be removed based on SHAP importance
    for current iteration of feature elimination:
    - The order of the features to keep is preserved.
    - The order of the features to remove is from worst to best feature.

    Args:
        shap_importance_df (pd.DataFrame):
            DataFrame containing SHAP importance values for features.

        step (Union[int, float]):
            The number of features to remove in each iteration.

        min_features_to_select (int):
            The minimum number of features to keep in the dataset.

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
    features_to_remove = _get_current_features_to_remove(
        shap_importance_df, step, min_features_to_select, columns_to_keep=columns_to_keep
    )

    # Convert features_to_remove to a set for O(1) lookup times
    features_to_remove_set = set(features_to_remove)

    # Filter out the features to remove, maintaining the original order of current_features_set
    remaining_features = [feature for feature in current_features_set if feature not in features_to_remove_set]

    return remaining_features, features_to_remove


def report_current_results(
    report_df: pd.DataFrame,
    round_number: int,
    current_features_set: List[str],
    features_to_remove: List[str],
    train_metric_mean: float,
    train_metric_std: float,
    val_metric_mean: float,
    val_metric_std: float,
) -> pd.DataFrame:
    """
    Records the results of the current feature elimination iteration.

    This method updates the report DataFrame with details about the current
    feature set, removed features, and model performance metrics.

    Args:
        report_df (pd.DataFrame):
            The DataFrame to store the results.

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

    Returns:
        pd.DataFrame:
            The updated report DataFrame with the current results.
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
    if report_df.empty:
        report_df = new_row
    else:
        report_df = pd.concat([report_df, new_row])

    return report_df


def get_best_num_features(
    report_df: pd.DataFrame,
    best_method: Literal["best", "best_coherent", "best_parsimonious"],
    standard_error_threshold: float = 1.0,
    verbose: Literal[0, 1, 2] = 0,
) -> int:
    """
    Determines the optimal number of features based on the specified selection strategy.

    Args:
        report_df (pd.DataFrame):
            The report DataFrame containing the feature elimination results.

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

        verbose (Literal[0, 1, 2], optional):
            The level of verbosity:
            - `0`: No logging.
            - `1`: Log the report.
            - `2`: Log the report and the best number of features.
            Default is `0`.

    Returns:
        int:
            The optimal number of features based on the selected strategy.

    Raises:
        ValueError:
            - If `best_method` is not one of `"best"`, `"best_coherent"`, or `"best_parsimonious"`.
            - If `standard_error_threshold` is negative or otherwise invalid.
    """
    if not isinstance(standard_error_threshold, (int, np.int64, float, np.float64)) or standard_error_threshold < 0:
        raise ValueError("Parameter standard_error_threshold must be a non-negative int or float.")

    # Create a copy of the report DataFrame to avoid modifying the original
    shap_report = report_df.copy()

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
        raise ValueError("The parameter 'best_method' must be one of 'best', 'best_coherent', or 'best_parsimonious'.")

    # Log the report if verbose
    if verbose > 1:
        logger.debug(shap_report)

    return best_num_features


def get_feature_names(report_df: pd.DataFrame, num_features: int) -> List[str]:
    """
    Retrieves the list of feature names corresponding to a specific number of selected features.

    Args:
        report_df (pd.DataFrame):
            The report DataFrame containing the feature elimination results.

        num_features (int):
            The number of features to retrieve.

    Returns:
        List[str]:
            A list of feature names corresponding to the specified number of features.

    Raises:
        ValueError:
            If the requested `num_features` was not achieved during the feature elimination process.
    """
    # Find the row in the report with the specified number of features
    matching_rows = report_df[report_df.num_features == num_features]

    if matching_rows.empty:
        valid_nums = ", ".join([str(n) for n in sorted(report_df.num_features.unique())])
        raise ValueError(
            f"The provided number of features ({num_features}) was not achieved during feature elimination. "
            f"Valid options are: {valid_nums}"
        )

    # Return the feature set from the first matching row
    return matching_rows.iloc[0]["features_set"]


def get_feature_support(column_names: List[str], feature_names_selected: List[str]) -> List[bool]:
    """
    Generates a boolean mask indicating which features were selected.

    Args:
        column_names (List[str]):
            A list of feature names that were selected after feature elimination.

        feature_names_selected (List[str]):
            A list of feature names that were selected after feature elimination.

    Returns:
        List[bool]:
            A boolean mask where `True` indicates a selected feature and `False` indicates a removed feature.
    """
    # Create a boolean mask where True indicates the feature was selected
    if column_names is None:
        raise ValueError("Feature names are not available. The model may not be fitted.")
    return [col in feature_names_selected for col in column_names]


def get_feature_ranking(report_df: pd.DataFrame, column_names: Optional[List[str]] = None) -> List[int]:
    """
    Retrieves the ranking of features based on their elimination order & metric score.

    The ranking is computed as follows:
    - Features that were never eliminated receive a rank of `1` (most important).
    - Features eliminated earlier receive higher ranks (indicating lower importance),
    while features that were never eliminated receive a rank of `1` (most important).

    Args:
        report_df (pd.DataFrame):
            The report DataFrame containing the feature elimination results.

        column_names (Optional[List[str]], optional):
            The names of the features in the original dataset.
            Default is `None`.

    Returns:
        List[int]:
            A list of feature rankings, where lower values indicate more important features
            and higher values correspond to features eliminated earlier.
    """
    if column_names is None:
        raise ValueError("Feature names are not available. The model may not be fitted.")

    # Get features that were never eliminated
    kept_features = report_df["features_set"].iloc[-1]

    # Get features that were eliminated (best to worst)
    eliminated_features = [
        eliminated_feature
        for eliminated_features_per_run in report_df["eliminated_features"]
        for eliminated_feature in eliminated_features_per_run
    ][::-1]

    # Create a dictionary of features and their ranking
    kept_features_ranking_dict = {feature: 1 for feature in kept_features}
    eliminated_features_ranking_dict = {feature: idx + 2 for idx, feature in enumerate(eliminated_features)}

    # Create the ranking list in the original column order
    ranking_dict = {**kept_features_ranking_dict, **eliminated_features_ranking_dict}

    # Return the ranking list in the original column order
    return [ranking_dict.get(col, 0) for col in column_names]
