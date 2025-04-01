from .explanation import calculate_shap_explanation
from .multiclass import aggregate_multiclass_shap_values_values, extract_multiclass_shap_parameters


import numpy as np
import pandas as pd
from shap import Explanation
from sklearn.base import BaseEstimator


import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


def process_shap_values(
    shap_explanation: Explanation,
    classes: Optional[List[Any]] = None,
    class_selection: Optional[Any] = None,
    weights: Optional[Dict[Any, float]] = None,
    multiclass_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Process SHAP values into a consistent format.

    This function handles multi-class SHAP values by either selecting a specific class
    or aggregating values across classes using one of the supported methods.

    Args:
        shap_explanation (Explanation):
            SHAP explanation object from the explainer.

        classes (Optional[List[Any]], optional):
            List of classes for multi-class models.

        class_selection (Optional[Any], optional):
            Only works when classes are provided (and thus are not None).

            For multi-class models only: class name or index to select SHAP values for.
            This extracts values for a single specific class instead of aggregating
            across classes. Ignored for binary classification or regression models.
            Default is None (no specific class selected).

        weights (Optional[Dict[Any, float]], optional):
            Only works when classes are provided (and thus are not None).

            Determines how to weight SHAP values across classes in multi-class scenarios:

            - dict: Dictionary with class names/indices as keys and weights as values for
              custom weighting. This allows for precise control over the importance of each class.

            The weighting is valuable in multi-class scenarios:
                - When certain classes are more critical to predict correctly, you can
                  emphasize their importance.
                - When classes are imbalanced, you can balance their influence on the overall
                  feature importance.
                - Example: For a 5-class problem with imbalanced classes, you might use
                  `{0: 0.1, 1: 0.3, 2: 0.1, 3: 0.4, 4: 0.1}` to prioritize classes 1 and 3.

            Default is None (no weighting). Note: This parameter is only applicable for
            multi-class models and has no effect on binary classification.

        multiclass_aggregation (Optional[Literal["max_abs", "mean", "mean_abs"]], optional):
            Method to aggregate SHAP values across classes for multi-class models:

            - "max_abs": Maximum absolute SHAP value across classes for each feature.
              This highlights features that are strongly influential for at least one class,
              which is useful for identifying the most discriminative features regardless
              of which class they affect most.

            - "mean": Mean SHAP value across classes.
              This provides a balanced measure of feature importance by averaging the
              contribution across all classes.

            - "mean_abs": Mean absolute SHAP value across classes.
              This provides the most balanced measure of feature importance by averaging
              the absolute contribution across all classes, preventing positive and negative
              contributions from canceling each other out.

            Default is None (uses first class for multi-class).

        shap_variance_penalty_factor (Optional[Union[int, float]], optional):
            Factor to penalize features with high variance in their SHAP values.
            Default is None (no penalty).

    Returns:
        np.ndarray:
            Processed SHAP values in a consistent format.
    """
    # Determine if the SHAP values are for a multi-class problem (3D) or binary/regression (2D)
    shap_values_shape = shap_explanation.values.shape
    is_multiclass = len(shap_values_shape) == 3

    # Handle binary classification case (2 classes, 3D SHAP values)
    if is_multiclass and shap_values_shape[2] == 2:
        # For binary classification, we return the SHAP values for the positive class
        # as it is equivalent to the SHAP values for the two classes
        return shap_explanation.values[:, :, 1]

    # Handle binary/regression case (2 or 1 class, 2D SHAP values)
    if not is_multiclass:
        if class_selection is not None:
            warnings.warn(
                "`class_selection` parameter is ignored for binary classification or regression models."
                " This parameter only applies to multi-class models.",
                UserWarning,
            )
        if weights is not None:
            warnings.warn(
                "`weights` parameter is ignored for binary classification or regression models."
                " This parameter only applies to multi-class models.",
                UserWarning,
            )
        if multiclass_aggregation is not None:
            warnings.warn(
                "`multiclass_aggregation` parameter is ignored for binary classification or regression models."
                " This parameter only applies to multi-class models.",
                UserWarning,
            )
        return shap_explanation.values

    # Get a copy of the original SHAP values to modify
    shap_values = shap_explanation.values.copy()

    # Convert numpy array to list if it is a numpy array
    classes = classes.tolist() if isinstance(classes, np.ndarray) else classes

    # Handle class selection - highest priority, returns immediately
    if class_selection is not None and classes is not None:
        return _get_shap_values_for_class(shap_explanation, class_selection, classes)

    # Apply class weighting if specified (modifies the original 3D values in place)
    if weights is not None and classes is not None:
        shap_values = _apply_class_weighting(shap_values, weights, classes)

    # Apply aggregation across classes if specified (reduces dimension to 2D)
    if multiclass_aggregation is not None:
        return aggregate_multiclass_shap_values_values(
            shap_values, multiclass_aggregation, shap_variance_penalty_factor
        )

    # If aggregation wasn't applied but we need to reduce dimensions:
    # - If weighting was applied, reduce by summing across classes
    # - If no weighting was applied, return the first class
    if weights is not None:
        # Sum weighted values across classes to get final 2D result
        return np.sum(shap_values, axis=2)
    else:
        # Default: return all multi-class SHAP values
        return shap_values


def calculate_shap_and_expected_values(
    model: BaseEstimator,
    X: pd.DataFrame,
    approximate: bool = False,
    verbose: Literal[0, 1, 2] = 0,
    random_state: Optional[int] = None,
    **shap_kwargs: Any,
) -> Tuple[Explanation, float]:
    """
    Prepare SHAP values and extract the expected value.

    This helper method calculates SHAP values and extracts the expected value.

    Args:
        model (BaseEstimator):
            The trained model to interpret. Must implement either predict or predict_proba
            method depending on the analysis requirements.

        X (pd.DataFrame):
            Feature data, of shape (n_samples, n_features).

        approximate (bool, default=False):
            If True, uses faster but less accurate SHAP calculation.

        verbose (Literal[0, 1, 2], optional):
            Controls the level of output messages:
            - `0`: No output or warnings.
            - `1`: Only important warnings.
            - `2`: All warnings and detailed logs.
            - Default is `0`.

        random_state (Optional[int], default=None):
            Random state for reproducibility.

        **shap_kwargs (Any):
            Additional arguments passed to SHAP Explainer, such as 'check_additivity'.

    Returns:
        Tuple[Explanation, float]:
            - SHAP explanation object
            - Expected value of the explainer
    """
    # Split arguments for multi-classification
    _, filtered_shap_kwargs = extract_multiclass_shap_parameters(shap_kwargs)

    # Calculate SHAP values for validation set
    shap_explanation, shap_explainer = calculate_shap_explanation(
        model,
        X,
        return_explainer=True,
        approximate=approximate,
        verbose=verbose,
        random_state=random_state,
        **filtered_shap_kwargs,
    )
    expected_value = shap_explainer.expected_value

    # For sklearn models, the expected value consists of n elements
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]

    return shap_explanation, expected_value


def shap_explanation_to_shap_values(
    shap_explanation: Explanation,
    model: BaseEstimator,
    X: pd.DataFrame,
    class_selection: Optional[Any] = None,
    multiclass_aggregation: Optional[Literal["mean", "max_abs", "mean_abs"]] = None,
    weights: Optional[Dict[Any, float]] = None,
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Process a SHAP explanation object, format the values, and convert to a DataFrame.

    This function processes SHAP values from an Explanation object, applying the
    appropriate formatting for multi-class models, and returns them as a pandas DataFrame.

    Args:
        shap_explanation (Explanation):
            SHAP explanation object from the explainer.

        model (BaseEstimator):
            Trained model used for the explanation.

        X (pd.DataFrame):
            Feature set used to calculate SHAP values.

        class_selection (Optional[Any], optional):
            For multi-class models only: class name or index to select SHAP values for.
            This extracts values for a single specific class instead of aggregating
            across classes. Ignored for binary classification or regression models.
            Default is None (no specific class selected).

        multiclass_aggregation (Optional[Literal["mean", "max_abs", "mean_abs"]], optional):
            Method to aggregate SHAP values across classes for multi-class models:

            - "mean": Mean SHAP value across classes.
              This provides a balanced measure of feature importance by averaging
              the SHAP values across all classes, preventing positive and negative
              contributions from canceling each other out.

            - "max_abs": Maximum absolute SHAP value across classes for each feature.
              This highlights features that are strongly influential for at least one class,
              which is useful for identifying the most discriminative features regardless
              of which class they affect most.

            - "mean_abs": Mean absolute SHAP value across classes.
              This provides the most balanced measure of feature importance by averaging
              the absolute contribution across all classes, preventing positive and negative
              contributions from canceling each other out.

            Default is None (uses first class for multi-class).

        weights (Optional[Dict[Any, float]], optional):
            Determines how to weight SHAP values across classes in multi-class scenarios:

            - dict: Dictionary with class names/indices as keys and weights as values for
              custom weighting. This allows for precise control over the importance of each class.

            The weighting is valuable in multi-class scenarios:
                - When certain classes are more critical to predict correctly, you can
                  emphasize their importance.
                - When classes are imbalanced, you can balance their influence on the overall
                  feature importance.
                - Example: For a 5-class problem with imbalanced classes, you might use
                  `{0: 0.1, 1: 0.3, 2: 0.1, 3: 0.4, 4: 0.1}` to prioritize classes 1 and 3.

            Default is None (no weighting). Note: This parameter is only applicable for
            multi-class models and has no effect on binary classification.

        shap_variance_penalty_factor (Optional[Union[int, float]], optional):
            Factor to penalize features with high variance in their SHAP values.
            Higher values favor features with more consistent impact across samples.
            Recommended values are between 0.5 and 1.0.
            Formula: penalized_mean = mean - (std * penalty_factor)
            If None or negative, no penalty is applied. Default is None.

    Returns:
        Union[pd.DataFrame, np.ndarray]:
            SHAP values in a DataFrame format with the same column names as the input data.
            If the model is multi-class, and no aggregation is performed the SHAP values
            are returned as a multi-dimensional array.
    """
    # Get the model classes. Regression models do not have classes.
    try:
        model_classes = model.classes_
    except AttributeError:
        model_classes = None

    # Format the SHAP values to a consistent format
    shap_values: np.ndarray = process_shap_values(
        shap_explanation=shap_explanation,
        classes=model_classes,
        class_selection=class_selection,
        weights=weights,
        multiclass_aggregation=multiclass_aggregation,
        shap_variance_penalty_factor=shap_variance_penalty_factor,
    )

    if shap_values.ndim == 3:
        warnings.warn(
            "Multi-class SHAP values detected. No aggregation was performed. Returning a multi-dimensional array instead."
        )
        return shap_values

    # Convert SHAP values to a pandas DataFrame; multi-class aggregation is done in the previous function
    return _shap_values_to_df(
        model=model,
        X=X,
        precalc_shap=shap_values,
        class_selection=None,
        weights=None,
        multiclass_aggregation=None,
        shap_variance_penalty_factor=None,
    )


def _apply_class_weighting(
    shap_values: np.ndarray,
    weights: Dict[Union[str, int], float],
    model_classes: Union[List[Union[str, int]], np.ndarray],
) -> np.ndarray:
    """
    Apply weighting to multi-class SHAP values while preserving the 3D structure.

    Args:
        shap_values (np.ndarray): Original 3D SHAP values

        weights (Dict[Union[str, int], float]): Type of weighting to apply
            - dict: Dictionary with class names as keys and weights as values
            if class names are int, they are regarded as class indices
            if class names are str, they are regarded as class names

        model_classes (Union[List[Union[str, int]], np.ndarray]): List of available class names/indices

    Returns:
        np.ndarray: Weighted SHAP values (still in 3D format)

    Raises:
        ValueError: If an unsupported weights is provided
    """
    # Convert numpy array or list to list[str]
    if isinstance(model_classes, (np.ndarray, list)):
        model_classes = list(map(str, model_classes))

    if isinstance(weights, dict):
        # If string keys: find the index of the class name
        if isinstance(list(weights.keys())[0], str):
            # Sort weights by class name
            weights = dict(sorted({model_classes.index(key): value for key, value in weights.items()}))
        # If int keys: use the weights as is
        else:
            # Sort weights by class index
            weights = dict(sorted(weights.items()))

        # Normalize the weights to sum up to 1 if they are non-zero
        if np.sum(list(weights.values())) > 0 and len(set(weights.values())) > 2:
            weights = {key: value / np.sum(list(weights.values())) for key, value in weights.items()}
        else:
            # If all weights are 0 or equal, return the original SHAP values
            return shap_values
    else:
        raise ValueError(f"Unsupported weights: {weights}. Use a dictionary of weights.")

    # Apply weights to each class in the 3D array (n_samples, n_features, n_classes)
    # Create weighted values array with the same shape as input
    weighted_values = np.zeros_like(shap_values)

    # Apply weights to each class (already sorted)
    for index, weight in enumerate(weights.values()):
        weighted_values[:, :, index] = shap_values[:, :, index] * weight

    return weighted_values


def _get_shap_values_for_class(
    shap_explanation: Explanation,
    class_selection: Union[str, int],
    model_classes: Union[List[Union[str, int]], np.ndarray],
) -> np.ndarray:
    """
    Extract SHAP values for a specific class.

    Model_classes is sorted by alphabetical order. Pay attention to this when selecting a class.

    Args:
        shap_explanation (Explanation): SHAP explanation object

        class_selection (Union[str, int]): Class name or index to select SHAP values for.
            If int: returns the SHAP values for the class at the given index.
            If str: returns the SHAP values for the class with the given name.

        model_classes (Union[List[Union[str, int]], np.ndarray]): List of available class names/indices

    Returns:
        np.ndarray: SHAP values for the specified class

    Raises:
        ValueError: If the requested class is not found
    """
    # Convert numpy array or list to list[str]
    if isinstance(model_classes, (np.ndarray, list)):
        model_classes = list(map(str, model_classes))

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
    return shap_explanation.values[:, :, class_idx]


def _shap_values_to_df(
    model: BaseEstimator,
    X: pd.DataFrame,
    precalc_shap: Optional[np.ndarray] = None,
    class_selection: Optional[Any] = None,
    multiclass_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
    weights: Optional[Dict[Any, float]] = None,
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convert SHAP values to a pandas DataFrame with the same structure as the input data.

    This function either uses precalculated SHAP values or calculates them using the
    provided model and dataset, then returns them in a DataFrame format that preserves
    the original feature names and index.

    Args:
        model (BaseEstimator):
            Pretrained model to explain.

        X (pd.DataFrame):
            Dataset on which the SHAP importance is calculated or was calculated.
            Used to get column names and index for the output DataFrame.

        precalc_shap (Optional[np.ndarray], optional):
            Precalculated SHAP values. If None, they are computed using the shap_calc function.
            Default is None.

        class_selection (Optional[Any], optional):
            For multi-class models only: class name or index to select SHAP values for.
            This extracts values for a single specific class instead of aggregating
            across classes. Ignored for binary classification or regression models.
            Default is None (no specific class selected).

        multiclass_aggregation (Optional[Literal["max_abs", "mean", "mean_abs"]], optional):
            Method to aggregate SHAP values across classes for multi-class models:

            - "max_abs": Maximum absolute SHAP value across classes for each feature.
              This highlights features that are strongly influential for at least one class,
              which is useful for identifying the most discriminative features regardless
              of which class they affect most.

            - "mean": Mean SHAP value across classes.
              This provides a balanced measure of feature importance by averaging the
              contribution across all classes.

            - "mean_abs": Mean absolute SHAP value across classes.
              This provides the most balanced measure of feature importance by averaging
              the absolute contribution across all classes, preventing positive and negative
              contributions from canceling each other out.

            Default is None (uses first class for multi-class).

        weights (Optional[Dict[Any, float]], optional):
            Determines how to weight SHAP values across classes in multi-class scenarios:

            - dict: Dictionary with class names/indices as keys and weights as values for
              custom weighting. This allows for precise control over the importance of each class.

            The weighting is valuable in multi-class scenarios:
                - When certain classes are more critical to predict correctly, you can
                  emphasize their importance.
                - When classes are imbalanced, you can balance their influence on the overall
                  feature importance.
                - Example: For a 5-class problem with imbalanced classes, you might use
                  `{0: 0.1, 1: 0.3, 2: 0.1, 3: 0.4, 4: 0.1}` to prioritize classes 1 and 3.

            Default is None (no weighting). Note: This parameter is only applicable for
            multi-class models and has no effect on binary classification.

        shap_variance_penalty_factor (Optional[Union[int, float]], optional):
            Penalty factor for the variance penalty. Default is None. Only used if
            multiclass_aggregation is not None.

        **shap_kwargs (Any):
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values multi-classification conversion - parameters like 'class_selection', 'multiclass_aggregation', and 'weights'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multi-class models.

    Returns:
        pd.DataFrame:
            DataFrame containing SHAP values with the same columns and index as X
            (if X is a DataFrame) or with generated column names (if X is a numpy array).

    Raises:
        ValueError: If input X is empty (has 0 rows or 0 columns).
        TypeError: If X is not a DataFrame or 2D array-like object.
    """
    # Use precalculated SHAP values if provided, otherwise calculate them
    if precalc_shap is not None:
        if len(precalc_shap) == 0:
            raise ValueError("Precalculated SHAP values are empty")

        shap_values = precalc_shap
    else:
        shap_explanation = calculate_shap_explanation(model, X, return_explainer=False, **kwargs)
        shap_values = shap_explanation_to_shap_values(
            shap_explanation=shap_explanation,
            model=model,
            X=X,
            class_selection=class_selection,
            multiclass_aggregation=multiclass_aggregation,
            weights=weights,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

    # Create DataFrame from SHAP values
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    elif (isinstance(X, np.ndarray) or hasattr(X, "shape")) and len(X.shape) == 2:
        # For numpy arrays, create generic column names
        return pd.DataFrame(shap_values)
    else:
        # If X is neither a DataFrame nor a 2D array-like object, raise an error
        raise TypeError("X must be a dataframe or a 2d array-like object")
