from probatus._wrapper import (
    calculate_base_shap_statistics,
    calculate_shap_explanation,
    calculate_shap_importance_dataframe,
    create_importance_dataframe,
    extract_multiclass_shap_parameters,
    shap_explanation_to_shap_values,
    process_shap_values,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV


from typing import Any, List, Literal, Tuple, Optional, Union


def create_shap_values(
    model: Union[BaseEstimator, BaseSearchCV],
    X_val: pd.DataFrame,
    score_train: float,
    score_val: float,
    verbose: Literal[0, 1, 2] = 0,
    random_state: Optional[int] = None,
    execution_mode: Literal["parallel", "vectorized"] = "vectorized",
    **shap_kwargs: Any,
) -> Tuple[Union[np.ndarray, dict[str, np.ndarray]], float, float]:
    """
    Processes SHAP values for the validation set based on the specified execution mode.

    This method handles the calculation and formatting of SHAP values, applying
    appropriate transformations based on whether vectorized or parallel execution
    is selected.

    Args:
        model (Union[BaseEstimator, BaseSearchCV]):
            The trained model to use for SHAP value calculation.

        X_val (pd.DataFrame):
            Validation dataset for SHAP value calculation.

        score_train (float):
            Model performance score on the training set.

        score_val (float):
            Model performance score on the validation set.

        verbose (Literal[0, 1, 2], optional):
            Verbosity level for logging.
            Default is `0`.

        random_state (Optional[int], optional):
            Random state for reproducibility.
            Default is `None`.

        execution_mode (Literal["parallel", "vectorized"]):
            The execution mode determining how SHAP values are processed:
            - "vectorized": Concatenates SHAP values across folds before aggregation.
            - "parallel": Aggregates SHAP values within each fold first.

        **shap_kwargs (Any):
            Additional keyword arguments for SHAP calculation and processing.

    Returns:
        Tuple[Union[np.ndarray, dict[str, np.ndarray]], float, float]:
            A tuple containing:
            - SHAP values or statistics for validation samples
            - Training score
            - Validation score
    """
    # Split arguments for multi-classification
    multi_class_kwargs, shap_kwargs = extract_multiclass_shap_parameters(shap_kwargs)

    # Calculate SHAP values for validation set
    shap_explanation_val = calculate_shap_explanation(
        model,
        X_val,
        return_explainer=False,
        verbose=verbose,
        random_state=random_state,
        **shap_kwargs,
    )

    if execution_mode == "vectorized":
        shap_values_val = shap_explanation_to_shap_values(
            shap_explanation=shap_explanation_val,
            model=model,
            X=X_val,
            **multi_class_kwargs,
        )

        return shap_values_val, score_train, score_val
    else:
        # Get SHAP variance penalty factor & remove from multi_class_kwargs
        shap_variance_penalty_factor = multi_class_kwargs.pop("shap_variance_penalty_factor", 0)

        # Format SHAP values for validation set (& perform aggregations if multi-class and provided)
        shap_values_val: np.ndarray = process_shap_values(shap_explanation=shap_explanation_val, **multi_class_kwargs)

        # Calculate base SHAP statistics
        shap_statistics: dict[str, np.ndarray] = calculate_base_shap_statistics(
            shap_values_val, shap_variance_penalty_factor
        )

        return shap_statistics, score_train, score_val


def process_shap_fold_values(
    results_per_fold: list[Tuple[Union[np.ndarray, dict[str, np.ndarray]], float, float]],
    execution_mode: Literal["parallel", "vectorized"],
    remaining_removeable_features: List[str],
    shap_variance_penalty_factor: float,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Processes the results from cross-validation folds to calculate SHAP feature importance.

    This method extracts SHAP values and performance scores from the cross-validation results,
    then calculates feature importance based on the specified execution mode.

    Two execution modes are supported:
    - "vectorized": Concatenates SHAP values from all folds before aggregation, then
        calculates importance. More efficient for smaller datasets with balanced folds.
    - "parallel": First aggregates SHAP statistics within each fold, then combines statistics
        across folds. Better reflects the fold structure and is more memory-efficient for
        large datasets, especially with unbalanced folds.

    Args:
        results_per_fold (list[Tuple[Union[np.ndarray, dict[str, np.ndarray]], float, float]]):
            List of tuples, each containing:
            - SHAP values or statistics from a CV fold
            - Training score for that fold
            - Validation score for that fold

        execution_mode (Literal["parallel", "vectorized"]):
            The mode for processing SHAP values across folds.

        remaining_removeable_features (List[str]):
            List of feature names that are currently active in the model.

        shap_variance_penalty_factor (float):
            Penalty factor applied to SHAP values with high variance.
            Formula: penalized_importance = mean_importance - (std_importance * factor)
            Used to reduce the influence of features with unstable importance.

    Returns:
        Tuple[pd.DataFrame, float, float]:
            A tuple containing:
            - DataFrame with SHAP importance for each feature
            - Mean training score across all folds
            - Mean validation score across all folds
    """
    if execution_mode == "vectorized":
        # Extract SHAP statistics & scores from results
        shap_values = np.concatenate([current_result[0] for current_result in results_per_fold], axis=0)
        scores_train = [current_result[1] for current_result in results_per_fold]
        scores_val = [current_result[2] for current_result in results_per_fold]

        # Calculate SHAP importance for features
        shap_importance_df = calculate_shap_importance_dataframe(
            shap_values,
            remaining_removeable_features,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )
    else:
        # Extract SHAP statistics & scores from results
        shap_fold_statistics = [current_result[0] for current_result in results_per_fold]
        scores_train = [current_result[1] for current_result in results_per_fold]
        scores_val = [current_result[2] for current_result in results_per_fold]

        # Aggregate SHAP statistics across folds
        aggregated_shap_statistics = _summarize_shap_statistics(shap_fold_statistics)

        # Create the output DataFrame with the calculated metrics
        shap_importance_df = create_importance_dataframe(
            aggregated_shap_statistics,
            remaining_removeable_features,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

    return shap_importance_df, float(np.mean(scores_train)), float(np.mean(scores_val))


def _summarize_shap_statistics(shap_fold_statistics: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Aggregates SHAP statistics across multiple cross-validation folds.

    This method combines SHAP statistics from different folds by:
    1. Collecting all statistics for each metric across folds
    2. Computing appropriate aggregations (mean or max) for each statistic type

    Args:
        shap_fold_statistics (list[dict[str, np.ndarray]]):
            List of dictionaries containing SHAP statistics for each fold.
            Each dictionary should contain the following keys:
            - 'shap_abs_mean': Mean absolute SHAP values
            - 'shap_abs_std': Standard deviation of absolute SHAP values
            - 'shap_abs_max': Maximum absolute SHAP values
            - 'shap_mean': Mean SHAP values
            - 'shap_std': Standard deviation of SHAP values

    Returns:
        dict[str, np.ndarray]:
            Dictionary containing aggregated SHAP statistics across all folds:
            - 'shap_abs_mean': Mean of absolute SHAP values across folds
            - 'shap_abs_std': Mean of standard deviations of absolute SHAP values
            - 'shap_abs_max': Maximum absolute SHAP values across folds
            - 'shap_mean': Mean of SHAP values across folds
            - 'shap_std': Mean of standard deviations of SHAP values

    Note:
        - For 'shap_abs_max', the maximum value across folds is used
        - For all other statistics, the mean across folds is used
    """
    # Initialize a dictionary to store the aggregated statistics
    aggregated_statistics = {}

    # Gather all the SHAP statistics for each fold
    for fold_stats in shap_fold_statistics:
        # Create a list of all the SHAP statistics for each fold
        for key, value in fold_stats.items():
            if key not in aggregated_statistics:
                aggregated_statistics[key] = []
            aggregated_statistics[key].append(value)

    # Calculate the mean and max for each aggregated statistic
    for key, values in aggregated_statistics.items():
        if key == "shap_abs_max":
            aggregated_statistics[key] = np.max(values, axis=0)
        else:
            aggregated_statistics[key] = np.mean(values, axis=0)

    return aggregated_statistics
