import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


def calculate_shap_importance_dataframe(
    shap_values: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_columns_suffix: Optional[str] = None,
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """
    Calculate feature importance metrics based on SHAP values.

    This function computes several importance metrics for each feature:

    - Mean absolute SHAP value: Average magnitude of feature impact across all samples
    - Mean SHAP value: Average directional impact of the feature
    - Maximum absolute SHAP value: Peak impact of the feature on any sample
    - Penalized mean absolute SHAP (optional): Mean absolute SHAP with a variance penalty

    The function can optionally apply a variance penalty to favor features with more
    consistent impact across samples.

    Args:
        shap_values (pd.DataFrame):
            Array of SHAP values with shape (n_samples, n_features).

        columns (List[str], optional):
            List of feature names corresponding to the columns in shap_values.
            Must have the same length as the feature dimension of shap_values.
            If None, the column names from the shap_values DataFrame are used.

        output_columns_suffix (str, optional):
            Suffix to be added to the end of column names in the output DataFrame.
            Useful when comparing multiple models. Default is "".

        shap_variance_penalty_factor (Union[int, float], optional):
            Factor to penalize features with high variance in their SHAP values.
            Higher values favor features with more consistent impact across samples.
            Recommended values are between 0.5 and 1.0.
            Formula: penalized_mean = mean - (std * penalty_factor)
            If None or negative, no penalty is applied. Default is None.

    Returns:
        pd.DataFrame:
            DataFrame containing importance metrics for each feature, sorted by importance.
            Columns include:
            - mean_abs_shap_value: Average magnitude of feature impact
            - mean_shap_value: Average directional impact
            - max_abs_shap_value: Maximum absolute impact of feature on any sample
            - penalized_mean_abs_shap_value: (If penalty applied) Mean with variance penalty

    Raises:
        ValueError: If the number of columns doesn't match the feature dimension in shap_values.
    """
    # Preprocess input data and validate inputs
    shap_values_array, feature_columns, suffix = _preprocess_shap_importance_inputs(
        shap_values, columns, output_columns_suffix
    )

    # Calculate base SHAP statistics
    shap_statistics = calculate_base_shap_statistics(shap_values_array, shap_variance_penalty_factor)

    # Create the output DataFrame with the calculated metrics
    importance_df = create_importance_dataframe(shap_statistics, feature_columns, suffix, shap_variance_penalty_factor)

    return importance_df


def create_importance_dataframe(
    shap_statistics: Dict[str, np.ndarray],
    columns: List[str],
    output_columns_suffix: str = "",
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """
    Create and format the feature importance DataFrame from calculated statistics.

    Args:
        shap_statistics: Dictionary of calculated SHAP statistics
        columns: Feature column names for the DataFrame index
        output_columns_suffix: Suffix for output column names
        shap_variance_penalty_factor: Variance penalty factor (for determining sort column)

    Returns:
        Formatted DataFrame with importance metrics, sorted by importance
    """
    # Create output dictionary for DataFrame
    df_dict: Dict[str, np.ndarray] = {
        f"mean_abs_shap_value{output_columns_suffix}": shap_statistics["shap_abs_mean"],
        f"mean_shap_value{output_columns_suffix}": shap_statistics["shap_mean"],
        f"max_abs_shap_value{output_columns_suffix}": shap_statistics["shap_abs_max"],
    }

    # Only include penalized values if a penalty factor was applied
    penalty_applied = shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0
    if penalty_applied and "penalized_shap_abs_mean" in shap_statistics:
        df_dict[f"penalized_mean_abs_shap_value{output_columns_suffix}"] = shap_statistics["penalized_shap_abs_mean"]

    # Create DataFrame with feature names as index
    importance_df = pd.DataFrame(df_dict, index=columns).astype(float)

    # Determine which column to use for sorting (penalized if available, otherwise regular mean abs)
    sort_column = (
        f"penalized_mean_abs_shap_value{output_columns_suffix}"
        if penalty_applied
        else f"mean_abs_shap_value{output_columns_suffix}"
    )

    # Sort by importance
    importance_df = importance_df.sort_values(sort_column, ascending=False)

    return importance_df


def _calculate_aggregated_binary_shap_statistics(
    shap_values: np.ndarray,
    abs_shap_values: np.ndarray,
    shap_variance_penalty_factor: float = 0,
) -> Dict[str, np.ndarray]:
    """
    Process SHAP statistics for binary classification or regression models.

    Args:
        shap_values: Raw SHAP values array (2D)
        abs_shap_values: Absolute SHAP values array (2D)
        shap_variance_penalty_factor: Variance penalty factor

    Returns:
        Dictionary of calculated statistics
    """
    # For binary classification or regression, calculate statistics directly
    shap_abs_mean = np.mean(abs_shap_values, axis=0)  # (n_features)
    shap_mean = np.mean(shap_values, axis=0)  # (n_features)

    # Calculate maximum absolute SHAP values for each feature
    shap_abs_max = np.max(abs_shap_values, axis=0)  # (n_features)

    # Initialize result dictionary
    result = {
        "shap_abs_mean": shap_abs_mean,
        "shap_mean": shap_mean,
        "shap_abs_max": shap_abs_max,
    }

    # Apply variance penalty if requested
    if shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0:
        penalized_shap_abs_mean = shap_abs_mean - (np.std(abs_shap_values, axis=0) * shap_variance_penalty_factor)
        result["penalized_shap_abs_mean"] = penalized_shap_abs_mean

    return result


def _calculate_aggregated_multiclass_shap_statistics(
    shap_values: np.ndarray,
    abs_shap_values: np.ndarray,
    shap_variance_penalty_factor: float = 0,
) -> Dict[str, np.ndarray]:
    """
    Process SHAP statistics for multi-class shap_values (n_samples, n_features, n_classes).

    Args:
        shap_values: Raw SHAP values array (3D)
        abs_shap_values: Absolute SHAP values array (3D)
        shap_variance_penalty_factor: Variance penalty factor

    Returns:
        Dictionary of calculated statistics
    """
    # For multi-class, first sum across classes, then calculate statistics
    sum_abs_shap = np.sum(abs_shap_values, axis=2)  # (n_samples, n_features)
    sum_shap = np.sum(shap_values, axis=2)  # (n_samples, n_features)

    # Calculate means across samples
    shap_abs_mean = np.mean(sum_abs_shap, axis=0)  # (n_features)
    shap_mean = np.mean(sum_shap, axis=0)  # (n_features)

    # Calculate maximum absolute SHAP values for each feature
    shap_abs_max = np.max(sum_abs_shap, axis=0)  # (n_features)

    # Initialize result dictionary
    result = {
        "shap_abs_mean": shap_abs_mean,
        "shap_mean": shap_mean,
        "shap_abs_max": shap_abs_max,
    }

    # Apply variance penalty if requested
    if shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0:
        penalized_shap_abs_mean = shap_abs_mean - (np.std(sum_abs_shap, axis=0) * shap_variance_penalty_factor)
        result["penalized_shap_abs_mean"] = penalized_shap_abs_mean

    return result


def calculate_base_shap_statistics(
    shap_values: np.ndarray,
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate the basic SHAP statistics needed for feature importance.

    Args:
        shap_values: Numpy array of SHAP values
        shap_variance_penalty_factor: Factor for variance penalization

    Returns:
        Dictionary containing calculated statistics:
            - shap_abs_mean: Mean absolute SHAP values
            - shap_mean: Mean SHAP values
            - penalized_shap_abs_mean: Penalized mean absolute SHAP values (if penalty applied)
    """
    # Normalize penalty factor
    penalty_factor = 0
    if shap_variance_penalty_factor is not None and shap_variance_penalty_factor >= 0:
        penalty_factor = shap_variance_penalty_factor

    # Calculate absolute SHAP values once only for speedup reasons
    abs_shap_values = np.abs(shap_values)

    # Handle multi-class case (when shap_values has more than 2 dimensions)
    if np.ndim(shap_values) > 2:  # multi-class case
        # Process multi-class SHAP values
        return _calculate_aggregated_multiclass_shap_statistics(shap_values, abs_shap_values, penalty_factor)
    else:
        # Process binary/regression SHAP values
        return _calculate_aggregated_binary_shap_statistics(shap_values, abs_shap_values, penalty_factor)


def _preprocess_shap_importance_inputs(
    shap_values: Union[pd.DataFrame, np.ndarray],
    columns: Optional[List[str]] = None,
    output_columns_suffix: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], str]:
    """
    Preprocess and validate the inputs for SHAP importance calculation.

    Args:
        shap_values: Input SHAP values as DataFrame or numpy array
        columns: Feature column names
        output_columns_suffix: Suffix for output column names

    Returns:
        Tuple containing:
            - numpy array of SHAP values
            - list of feature column names
            - normalized output column suffix

    Raises:
        ValueError: If columns don't match SHAP value dimensions
    """
    # Convert to numpy array if DataFrame
    shap_values_array = shap_values.values if isinstance(shap_values, pd.DataFrame) else shap_values

    # Set columns appropriately
    feature_columns = shap_values.columns if columns is None else columns

    # Normalize the suffix
    suffix = "" if output_columns_suffix is None else output_columns_suffix

    # Validate dimensions
    if np.ndim(shap_values_array) >= 2:
        feature_dim = shap_values_array.shape[1]
        if len(feature_columns) != feature_dim:
            raise ValueError(
                f"Dimension mismatch: shap_values has {feature_dim} features but columns list has {len(feature_columns)} elements"
            )

    return shap_values_array, feature_columns, suffix
