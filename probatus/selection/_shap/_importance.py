from typing import Union, Optional, List, Tuple
import pandas as pd
import numpy as np


def _filter_and_identify_features_based_on_importance(
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
