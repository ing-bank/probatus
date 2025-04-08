import warnings
from probatus.wrapper.shap_new.instance import SHAPInstance


import numpy as np


from typing import Any, Dict, Literal, Optional, Union, List


def calculate_aggregated_values(
    shap_instance: SHAPInstance,
    class_selection: Optional[Any] = None,
    weights: Optional[Dict[Any, float]] = None,
    multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
    shap_variance_penalty_factor: Optional[Union[int, float]] = 0,
) -> np.ndarray:
    """
    Calculate aggregated SHAP values from a SHAP instance.

    Args:
        shap_instance (SHAPInstance): SHAP instance object
        class_selection (Optional[Any], optional): Class name or index to select SHAP values for.
            If int: returns the SHAP values for the class at the given index.
            If str: returns the SHAP values for the class with the given name.
            Defaults to None.
        weights (Optional[Dict[Any, float]], optional): Dictionary with class names or indices as keys and weights as values.
            Defaults to None.
        multi_class_aggregation (Optional[Literal[&quot;max_abs&quot;, &quot;mean&quot;, &quot;mean_abs&quot;]], optional): Aggregation method.
            Defaults to None.
        shap_variance_penalty_factor (Optional[Union[int, float]], optional): Variance penalty factor.
            Defaults to 0.

    Returns:
        np.ndarray: Aggregated SHAP values (or SHAP values for a single class)
    """
    # Determine if the SHAP values are for a multi-class problem (3D) or binary/regression (2D)
    shap_values_shape = shap_instance.values.shape
    is_multi_class = len(shap_values_shape) == 3

    # Handle binary classification case (2 classes, 3D SHAP values)
    if is_multi_class and shap_values_shape[2] == 2:
        warnings.warn(
            "The provided SHAP instance is a binary/regression SHAP instance, therefore no aggregation is applied.",
            UserWarning,
        )
        # For binary classification, we return the SHAP values for the positive class
        # as it is equivalent to the SHAP values for the two classes
        return shap_instance.values[:, :, 1]

    # Handle binary/regression case (2 or 1 class, 2D SHAP values)
    if not is_multi_class:
        if class_selection is not None:
            warnings.warn(
                "`class_selection`, `weights`, `multi_class_aggregation` parameters are ignored for binary/regression SHAP instances."
                " These parameters only apply to multi-class models.",
                UserWarning,
            )
        return shap_instance.values

    # Get properties from SHAP instance
    shap_values = shap_instance.values
    classes = shap_instance.explainer.output_names.tolist()

    if class_selection is not None:
        return _get_shap_values_for_class(shap_instance, class_selection, classes)

    # Apply class weighting if specified (modifies the original 3D values in place)
    if weights is not None:
        shap_values = _apply_class_weighting(shap_values, weights, classes)

    # Apply aggregation across classes if specified (reduces dimension to 2D)
    if multi_class_aggregation is not None:
        return aggregate_multiclass_shap_values_values(
            shap_values, multi_class_aggregation, shap_variance_penalty_factor
        )

    # Default: sum all class value
    return np.sum(shap_values, axis=2)


def _get_shap_values_for_class(
    shap_instance: SHAPInstance,
    class_selection: Union[str, int],
    model_classes: List[Union[str, int]],
) -> np.ndarray:
    """
    Extract SHAP values for a specific class.

    Args:
        shap_instance (SHAPInstance): SHAP instance object

        class_selection (Union[str, int]): Class name or index to select SHAP values for.
            If int: returns the SHAP values for the class at the given index.
            If str: returns the SHAP values for the class with the given name.

        model_classes (List[Union[str, int]]): List of available class names/indices

    Returns:
        np.ndarray: SHAP values for the specified class

    Raises:
        ValueError: If the requested class is not found
    """
    # Validate that the requested class exists
    if isinstance(class_selection, str) and class_selection not in model_classes:
        raise ValueError(f"Class '{class_selection}' not found in model classes: {model_classes}")

    # Handle different types of class selection
    if isinstance(class_selection, (int, np.int64)):
        class_idx = int(class_selection)
    elif isinstance(class_selection, str):
        class_idx = model_classes.index(class_selection)
    else:
        raise ValueError(f"Unsupported class selection type: {type(class_selection)}. Use 'str' or 'int'.")

    # Return SHAP values for the specified class
    return shap_instance.values[:, :, class_idx]


def _apply_class_weighting(
    shap_instance: SHAPInstance,
    weights: Dict[Union[str, int], float],
    model_classes: List[Union[str, int]],
) -> np.ndarray:
    """
    Apply weighting to multi-class SHAP values preserving the 3D structure.

    Args:
        shap_instance (SHAPInstance): Original SHAP instance
        weights (Dict[Union[str, int], float]): Dictionary with class names or indices as keys and weights as values
        model_classes (List[Union[str, int]]): List of available class names/indices

    Returns:
        np.ndarray: Weighted SHAP values (3D)

    Raises:
        ValueError: If weights format is unsupported
    """
    if not isinstance(weights, dict):
        raise ValueError(f"Unsupported weights: {weights}. Provide a dictionary.")

    # Convert string keys to indices if necessary
    weights = {(model_classes.index(k) if isinstance(k, str) else k): v for k, v in weights.items()}

    # Normalize weights if sum is positive and values differ
    weight_values = np.array(list(weights.values()))
    if weight_values.sum() > 0 and len(set(weight_values)) > 1:
        weight_values /= weight_values.sum()
    elif weight_values.sum() == 0:
        return shap_instance.values

    # Create a weights vector matching the class dimension
    weight_vector = np.zeros(shap_instance.values.shape[2])
    for idx, w in weights.items():
        weight_vector[idx] = w

    # Apply weights directly using broadcasting
    weighted_values = shap_instance.values * weight_vector[np.newaxis, np.newaxis, :]

    return weighted_values


def aggregate_multiclass_shap_values_values(
    shap_values: np.ndarray,
    aggregation_method: Literal["mean", "max_abs", "mean_abs"],
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Aggregate SHAP values across classes using the specified method.

    Args:
        shap_values (np.ndarray): SHAP values (n_samples, n_features, n_classes)
        aggregation_method (Literal["mean", "max_abs", "mean_abs"]): Aggregation method
        shap_variance_penalty_factor (Optional[Union[int, float]], optional):
            Variance penalty factor. Default is None.

    Returns:
        np.ndarray: Aggregated SHAP values (n_samples, n_features)

    Raises:
        ValueError: If unsupported aggregation method is provided
    """
    aggregation_methods = {
        "mean": lambda x: np.mean(x, axis=2),
        "max_abs": lambda x: np.max(np.abs(x), axis=2),
        "mean_abs": lambda x: np.mean(np.abs(x), axis=2),
    }

    if aggregation_method not in aggregation_methods:
        raise ValueError(
            f"Unsupported aggregation method: {aggregation_method}. Choose from {list(aggregation_methods.keys())}."
        )

    aggregated_shap = aggregation_methods[aggregation_method](shap_values)

    if shap_variance_penalty_factor is not None and shap_variance_penalty_factor > 0:
        variance_penalty = np.std(shap_values, axis=2) * shap_variance_penalty_factor
        aggregated_shap -= variance_penalty

    return aggregated_shap
