import numpy as np
from loguru import logger
import pandas as pd


from typing import List, Literal, Optional


def _get_best_num_features(
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


def _get_feature_names(report_df: pd.DataFrame, num_features: int, strict: bool = False) -> List[str]:
    """
    Retrieves the list of feature names corresponding to a specific number of selected features.
    If the exact number of features is not available, selects the closest number of features.
    When equally close options exist, the one with fewer features is selected.

    Args:
        report_df (pd.DataFrame):
            The report DataFrame containing the feature elimination results.

        num_features (int):
            The target number of features to retrieve.

        strict (bool, optional):
            If `True`, the function will raise an error if the exact number of features is not available.
            Default is `False`.

    Returns:
        List[str]:
            A list of feature names corresponding to the specified or closest number of features.

    Raises:
        ValueError:
            If the report DataFrame is empty.
    """
    # Check if report_df is empty
    if report_df.empty:
        raise ValueError("The report DataFrame is empty. The model may not be fitted.")

    # Find the row in the report with the specified number of features
    matching_rows = report_df[report_df.num_features == num_features]

    if not matching_rows.empty:
        # Return the feature set from the first matching row if exact match found
        return matching_rows.iloc[0]["features_set"]
    else:
        if strict:
            raise ValueError(
                f"The exact number of features ({num_features}) was not achieved during feature elimination."
            )
        else:
            # If no exact match, find the closest number of features
            available_features = sorted(report_df.num_features.unique())

            # Calculate absolute differences
            differences = [abs(n - num_features) for n in available_features]

            # Find minimum difference
            min_difference = min(differences)

            # Get all indices with minimum difference
            min_diff_indices = [i for i, diff in enumerate(differences) if diff == min_difference]

            # If multiple options with same difference, prefer the one with fewer features
            selected_feature_counts = [available_features[i] for i in min_diff_indices]
            selected_feature_count = min(selected_feature_counts)

            # Return the feature set for the selected number of features
            matching_rows = report_df[report_df.num_features == selected_feature_count]

            # Log a warning that we're using an approximation
            logger.warning(
                f"The provided number of features ({num_features}) was not achieved during feature elimination. "
                f"Using {selected_feature_count} features instead as the closest available option."
            )

            return matching_rows.iloc[0]["features_set"]


def _get_feature_ranking(report_df: pd.DataFrame, column_names: Optional[List[str]] = None) -> List[int]:
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


def _get_feature_support(column_names: List[str], feature_names_selected: List[str]) -> List[bool]:
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


def _report_current_results(
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
