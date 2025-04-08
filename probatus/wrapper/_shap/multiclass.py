import numpy as np
import warnings

from typing import Any, Dict, Literal, Optional, Tuple, Union


def aggregate_multiclass_shap_values_values(
    shap_values: np.ndarray,
    aggregation_method: Literal["mean", "max_abs", "mean_abs"],
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Aggregate SHAP values across classes using the specified method.

    Args:
        shap_values (np.ndarray): 3D SHAP values to aggregate (n_samples, n_features, n_classes)
        aggregation_method (Literal["mean", "max_abs", "mean_abs"]):
            Method to use for aggregation
        shap_variance_penalty_factor (Optional[Union[int, float]], optional):
            Factor to penalize features with high variance in their SHAP values.
            Default is None (no penalty).

    Returns:
        np.ndarray: Aggregated SHAP values or penalized SHAP values (2D: n_samples, n_features)

    Raises:
        ValueError: If an unsupported aggregation method is provided
    """
    if aggregation_method == "mean":
        # Mean SHAP value across classes
        mean_shap = np.mean(shap_values, axis=2)

        # Apply variance penalty if requested
        if shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0:
            penalized_mean_shap = mean_shap - (np.std(shap_values, axis=2) * shap_variance_penalty_factor)
            return penalized_mean_shap
        else:
            return mean_shap

    elif aggregation_method == "max_abs":
        # Maximum absolute SHAP value across classes for each feature
        max_abs_shap = np.max(np.abs(shap_values), axis=2)

        # Apply variance penalty if requested
        if shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0:
            penalized_max_abs_shap = max_abs_shap - (np.std(shap_values, axis=2) * shap_variance_penalty_factor)
            return penalized_max_abs_shap
        else:
            return max_abs_shap

    elif aggregation_method == "mean_abs":
        # Mean absolute SHAP value across classes
        mean_abs_shap = np.mean(np.abs(shap_values), axis=2)

        # Apply variance penalty if requested
        if shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0:
            penalized_mean_abs_shap = mean_abs_shap - (np.std(shap_values, axis=2) * shap_variance_penalty_factor)
            return penalized_mean_abs_shap
        else:
            return mean_abs_shap

    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}. Use 'mean', 'max_abs' or 'mean_abs'.")


# TODO: Remove since its duplicate (update code)
def extract_shap_parameters(
    shap_kwargs: Dict[str, Any],
    verbose: Literal[0, 1, 2] = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Extract parameters related to multi-class SHAP value conversions from shap_kwargs.

    This helper function separates parameters that are passed to the SHAP explainer
    from parameters that control multi-class SHAP value conversion.

    Args:
        shap_kwargs (Dict[str, Any]):
            Dictionary of keyword arguments for SHAP and multi-class processing.

        default_aggregation_method (Optional[Literal["mean", "max_abs", "mean_abs"]], optional):
            Default aggregation method to use if not specified in shap_kwargs.
            Default is None (no default aggregation method).

        verbose (Literal[0, 1, 2], optional):
            Verbosity level for logging.
            Default is 0 (no logging).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - First dict: Parameters for multi-class SHAP values conversion
            - Second dict: Parameters for SHAP explainer
    """
    # Parameters that are used only for multi-class SHAP value conversion
    default_multi_class_params = {
        "class_selection": None,
        "multiclass_aggregation": None,
        "weights": None,
        "shap_variance_penalty_factor": None,
    }

    # Extract parameters related to multi-class conversion
    multi_class_params = {}
    shap_explanation_kwargs = {}
    shap_explainer_kwargs = shap_kwargs.copy()

    # Multi-class parameters
    for param_name in default_multi_class_params:
        if param_name in shap_explainer_kwargs:
            multi_class_params[param_name] = shap_explainer_kwargs.pop(param_name)

    # SHAP explanation parameters
    for param_name in shap_explainer_kwargs:
        if param_name == "approximate":
            shap_explanation_kwargs[param_name] = shap_explainer_kwargs.pop(param_name)

    if verbose > 0:
        if multi_class_params.keys() == default_multi_class_params.keys():
            warnings.warn(
                "No multi-class parameters provided. Default values passed to the SHAP explainer.",
                UserWarning,
            )

    return multi_class_params, shap_explainer_kwargs, shap_explanation_kwargs
