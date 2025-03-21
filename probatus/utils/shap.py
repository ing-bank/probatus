import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from shap import Explainer, Explanation
from shap.explainers import TreeExplainer
from shap.utils import sample
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from probatus.utils.common import get_pipeline_preprocessor_and_estimator, preprocess_using_pipeline


def calculate_shap_explanation(
    model: BaseEstimator,
    X: pd.DataFrame,
    return_explainer: bool = False,
    verbose: Literal[0, 1, 2] = 0,
    random_state: Optional[int] = None,
    sample_size: int = 100,
    approximate: bool = False,
    check_additivity: bool = False,
    **shap_kwargs: Any,
) -> Union[Tuple[Explanation], Tuple[Explanation, Explainer]]:
    """
    Calculate the SHAP (SHapley Additive exPlanations) values for a given model.

    SHAP values help explain the output of machine learning models by attributing
    the prediction to each input feature.

    Probatus supports the following SHAP explainers:
    - TreeExplainer: for tree-based models
    - LinearExplainer: for linear models (e.g. LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)

    Not supported (for now):
    - KernelExplainer: for non-linear models (very slow -- so not intended to be used)
    - DeepExplainer: for deep learning models (not in line with the rest of the library)
    - SamplingExplainer: for models that do not support SHAP (not intended to be used)

    Args:
        model (BaseEstimator):
            Trained model to explain. Should be compatible with the SHAP library.

        X (pd.DataFrame):
            Feature set used to calculate SHAP values. Should have the same format
            as the data used to train the model.

        return_explainer (bool, optional):
            If True, returns the SHAP explainer.
            Default is False.

        verbose (Literal[0, 1, 2], optional):
            Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
            Default is `0`.

        random_state (int, optional):
            Random state for reproducibility when sampling background data.
            If None, results may not be reproducible. Default is None.

        sample_size (int, optional):
            Number of samples to use for creating the background dataset (masker).
            A sensible default is 100 samples as it provides a good balance between
            capturing the data's variability (which is critical for reliable SHAP values
            and additivity checks) and maintaining reasonable computation times.
            For smaller datasets, using around 20% of the data might be more appropriate,
            while for larger datasets, 100–500 samples may be used.

        approximate (bool, optional):
            If True, uses SHAP approximations for tree-based models—this is less accurate but much faster.
            Only applies to tree-based explainers. Default is False.

        check_additivity (bool, optional):
            If True, performs an additivity check to ensure the SHAP values sum to the model's prediction.
            Using a masker that accurately captures data variability helps avoid potential additivity
            issues. Default is False.

        **shap_kwargs:
            Additional keyword arguments passed to the SHAP Explainer.

    Returns:
        Union[Tuple[Explanation], Tuple[Explanation, Explainer]]:
            SHAP Explanation for the model, or a tuple (shap_explanation, explainer) if return_explainer is True.

    Raises:
        TypeError: If the model is not compatible with SHAP.
    """
    # Check prerequisites
    is_valid, error_message = _validate_shap_inputs(model, X, verbose)
    if not is_valid:
        raise TypeError(error_message)

    # Extract estimator and preprocess data if it's a pipeline
    original_X = X
    if isinstance(model, Pipeline):
        # Get the preprocessor and estimator from the pipeline
        preprocessor, estimator = get_pipeline_preprocessor_and_estimator(model)

        if preprocessor is not None and verbose > 1:
            warnings.warn("Applying preprocessing steps from pipeline before calculating SHAP values.")

        # Apply preprocessing to X if a preprocessor exists
        X = preprocess_using_pipeline(X, model)
        model = estimator

        if verbose > 1:
            warnings.warn(f"Using final estimator from pipeline: {type(model).__name__}")

    # Create the SHAP explainer
    explainer = _create_shap_explainer(
        model=model, X=X, random_state=random_state, sample_size=sample_size, **shap_kwargs
    )

    # Calculate SHAP values
    shap_explanation = _compute_shap_values(
        explainer=explainer, X=X, approximate=approximate, check_additivity=check_additivity
    )

    # Return the SHAP values, and the explainer for reuse
    if return_explainer:
        return shap_explanation, explainer
    # Return the SHAP explanation and the explainer
    return shap_explanation


def shap_explanation_to_shap_df(
    shap_explanation: Explanation,
    model: BaseEstimator,
    X: pd.DataFrame,
    class_selection: Optional[Any] = None,
    multiclass_aggregation: Optional[Literal["max_abs", "variance", "mean_abs"]] = None,
    weight_type: Optional[Union[Literal["frequency"], Dict[Any, float]]] = None,
) -> pd.DataFrame:
    """
    Process a SHAP explanation object, format the values, and convert to a DataFrame.

    This function processes SHAP values from an Explanation object, applying the
    appropriate formatting for multiclass models, and returns them as a pandas DataFrame.

    Args:
        shap_explanation (Explanation):
            SHAP explanation object from the explainer.

        model (BaseEstimator):
            Trained model used for the explanation.

        X (pd.DataFrame):
            Feature set used to calculate SHAP values.

        class_selection (Optional[Any], optional):
            For multiclass models only: class name or index to select SHAP values for.
            This extracts values for a single specific class instead of aggregating
            across classes. Ignored for binary classification or regression models.
            Default is None (no specific class selected).

        multiclass_aggregation (Optional[Literal["max_abs", "variance", "mean_abs"]], optional):
            Method to aggregate SHAP values across classes for multiclass models:

            - "max_abs": Maximum absolute SHAP value across classes for each feature.
              This highlights features that are strongly influential for at least one class,
              which is useful for identifying the most discriminative features regardless
              of which class they affect most.

            - "variance": Variance of SHAP values across classes.
              This identifies features whose impact varies significantly between classes,
              which can help detect features that play different roles in differentiating
              various classes.

            - "mean_abs": Mean absolute SHAP value across classes.
              This provides the most balanced measure of feature importance by averaging
              the absolute contribution across all classes, preventing positive and negative
              contributions from canceling each other out.

            Default is None (uses first class for multiclass).

        weight_type (Optional[Union[Literal["frequency"], Dict[Any, float]]], optional):
            Determines how to weight SHAP values across classes in multiclass scenarios:

            - 'frequency': Uses equal weights for all classes (1/n_classes). This is useful
              when all classes are equally important regardless of their frequency in the data.

            - dict: Dictionary with class names/indices as keys and weights as values for
              custom weighting. This allows for precise control over the importance of each class.

            The weighting is valuable in multiclass scenarios:
                - When certain classes are more critical to predict correctly, you can
                  emphasize their importance.
                - When classes are imbalanced, you can balance their influence on the overall
                  feature importance.
                - Example: For a 5-class problem with imbalanced classes, you might use
                  `{0: 0.1, 1: 0.3, 2: 0.1, 3: 0.4, 4: 0.1}` to prioritize classes 1 and 3.

            Default is None (no weighting). Note: This parameter is only applicable for
            multiclass models and has no effect on binary classification.

    Returns:
        pd.DataFrame:
            SHAP values in a DataFrame format with the same column names as the input data.
    """
    # Format the SHAP values to a consistent format
    shap_values: np.ndarray = _format_shap_values(
        shap_explanation=shap_explanation,
        class_selection=class_selection,
        multiclass_aggregation=multiclass_aggregation,
        weight_type=weight_type,
    )

    # Convert SHAP values to a pandas DataFrame
    return _shap_values_to_df(model=model, X=X, precalc_shap=shap_values)


# TODO: Expand this to support more optimization objectives
# For example:
# - Max SHAP contribution
# - Variance-based aggregation
# - Class-specific optimization
# - Feature-specific optimization
def calculate_shap_importance(
    shap_values: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_columns_suffix: Optional[str] = None,
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """
    Calculate feature importance metrics based on SHAP values.

    This function computes the average SHAP value and average absolute SHAP value
    for each feature, optionally applying a variance penalty to favor features
    with more consistent impact.

    Args:
        shap_values (pd.DataFrame):
            Array of SHAP values with shape (n_samples, n_features) or
            (n_classes, n_samples, n_features) for multi-class problems.

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
            Columns include mean absolute SHAP value, mean SHAP value, and optionally
            penalized mean absolute SHAP value if a penalty factor was applied.

    Raises:
        ValueError: If the number of columns doesn't match the feature dimension in shap_values.
    """
    # Initialize variables
    shap_values = shap_values.values if isinstance(shap_values, pd.DataFrame) else shap_values
    columns = shap_values.columns if columns is None else columns
    output_columns_suffix = "" if output_columns_suffix is None else output_columns_suffix

    # Check for dimension mismatch between shap_values and columns
    if np.ndim(shap_values) >= 2:
        feature_dim = shap_values.shape[1]
        if len(columns) != feature_dim:
            raise ValueError(
                f"Dimension mismatch: shap_values has {feature_dim} features but columns list has {len(columns)} elements"
            )

    # Validate and normalize the variance penalty factor
    if shap_variance_penalty_factor is None or shap_variance_penalty_factor < 0:
        # If None or negative, don't apply any penalty
        shap_variance_penalty_factor = 0

    # Calculate absolute SHAP values for magnitude-based importance
    abs_shap_values = np.abs(shap_values)

    # Handle multi-class case (when shap_values has more than 2 dimensions)
    if np.ndim(shap_values) > 2:  # multi-class case
        # For multi-class, first sum across classes, then calculate statistics
        sum_abs_shap = np.sum(abs_shap_values, axis=0)  # Sum across classes
        sum_shap = np.sum(shap_values, axis=0)  # Sum raw values across classes

        # Calculate means across samples
        shap_abs_mean = np.mean(sum_abs_shap, axis=0)
        shap_mean = np.mean(sum_shap, axis=0)

        # Apply variance penalty if requested
        penalized_shap_abs_mean = shap_abs_mean - (np.std(sum_abs_shap, axis=0) * shap_variance_penalty_factor)
    else:
        # For binary classification or regression, calculate statistics directly
        shap_abs_mean = np.mean(abs_shap_values, axis=0)
        shap_mean = np.mean(shap_values, axis=0)
        penalized_shap_abs_mean = shap_abs_mean - (np.std(abs_shap_values, axis=0) * shap_variance_penalty_factor)

    # Create a dictionary for the DataFrame columns
    # Always include mean absolute and raw SHAP values
    df_dict: Dict[str, np.ndarray] = {
        f"mean_abs_shap_value{output_columns_suffix}": shap_abs_mean,
        f"mean_shap_value{output_columns_suffix}": shap_mean,
    }

    # Only include penalized values if a penalty factor was applied
    if shap_variance_penalty_factor > 0:
        df_dict[f"penalized_mean_abs_shap_value{output_columns_suffix}"] = penalized_shap_abs_mean

    # Create DataFrame with feature names as index
    importance_df = pd.DataFrame(df_dict, index=columns).astype(float)

    # Determine which column to use for sorting (penalized if available, otherwise regular mean abs)
    sort_column = (
        f"penalized_mean_abs_shap_value{output_columns_suffix}"
        if shap_variance_penalty_factor > 0
        else f"mean_abs_shap_value{output_columns_suffix}"
    )

    importance_df = importance_df.sort_values(sort_column, ascending=False)

    return importance_df


def _validate_shap_inputs(
    model: BaseEstimator,
    X: pd.DataFrame,
    verbose: Literal[0, 1, 2] = 0,
) -> Tuple[bool, Optional[str]]:
    """
    Check prerequisites for SHAP calculation and validate inputs.

    Args:
        model (BaseEstimator):
            Trained model to explain. Should be compatible with SHAP library.

        X (pd.DataFrame):
            Feature set used to calculate SHAP values.

        verbose (Literal[0, 1, 2], optional):
                Controls the level of output messages:
                - `0`: No output or warnings.
                - `1`: Only important warnings.
                - `2`: All warnings and detailed logs.
                - Default is `0`.

    Returns:
        Tuple[bool, Optional[str]]:
            A tuple containing (is_valid, error_message).
            If is_valid is False, error_message contains the reason.
    """
    # SHAP doesn't work with scikit-learn pipelines
    if isinstance(model, Pipeline):
        if verbose > 0:
            warnings.warn(
                "The provided model is a Pipeline. SHAP does not directly support pipelines. "
                "Automatically extracting the final estimator from the pipeline."
            )

    # Check if the dataset is a pandas DataFrame
    if not isinstance(X, pd.DataFrame) and verbose > 0:
        warnings.warn("The provided dataset is not a pandas DataFrame. Categorical features are not recognized.")

    return True, None


def _create_shap_explainer(
    model: BaseEstimator,
    X: pd.DataFrame,
    random_state: Optional[int] = None,
    sample_size: int = 100,
    **shap_kwargs: Any,
) -> Explainer:
    """
    Create an appropriate SHAP explainer for the given model and data.

    Args:
        model (BaseEstimator):
            Trained model to explain.

        X (pd.DataFrame):
            Feature set used to calculate SHAP values.

        random_state (int, optional):
            Random state for reproducibility. Default is None.

        sample_size (int, optional):
            Number of samples to use for background data. Default is 100.

        **shap_kwargs:
            Additional keyword arguments passed to the SHAP Explainer.

    Returns:
        Explainer:
            Initialized SHAP explainer object.
    """
    # Check if the dataset has categorical features
    has_categorical = False
    if isinstance(X, pd.DataFrame):
        has_categorical = X.select_dtypes("category").shape[1] > 0

    # For non-tree models, we reserve a background sample (masker) to provide a realistic baseline for
    # perturbing features, since tree-based models inherently manage feature variations. Perturbing features
    # quantify each feature's contribution relative to this baseline, which is key to SHAP calculations.
    masker = None
    if not has_categorical and shap_kwargs.get("feature_perturbation") != "tree_path_dependent":
        # If the dataset is smaller than the requested sample size, use a percentage of the dataset
        if X.shape[0] < sample_size:
            sample_size = int(np.ceil(X.shape[0] * 0.2))  # Use 20% of the dataset

        # Create background data by sampling from the input dataset
        masker = sample(X, sample_size, random_state=random_state)

    # Initialize the SHAP explainer with the model and masker
    return Explainer(model, masker=masker, seed=random_state, **shap_kwargs)


def _compute_shap_values(
    explainer: Explainer,
    X: pd.DataFrame,
    approximate: bool = False,
    check_additivity: bool = True,
) -> Explanation:
    """
    Calculate SHAP values using the appropriate method based on explainer type.

    Args:
        explainer (Explainer):
            Initialized SHAP explainer.

        X (pd.DataFrame):
            Feature set to calculate SHAP values for.

        approximate (bool, optional):
            If True, uses SHAP approximations for tree-based models. Default is False.

        check_additivity (bool, optional):
            If False, disables additivity check for tree models. Default is True.

    Returns:
        Explanation:
            SHAP values.
    """
    # Calculate SHAP values based on the explainer type
    if isinstance(explainer, TreeExplainer):
        # Tree-based models can use approximation for faster calculation
        return explainer(X, check_additivity=check_additivity, approximate=approximate)
    else:
        # Standard calculation for non-tree models
        return explainer(X)


def _format_shap_values(
    shap_explanation: Explanation,
    class_selection: Optional[Any] = None,
    weight_type: Optional[Union[Literal["frequency"], Dict[Any, float]]] = None,
    multiclass_aggregation: Optional[Literal["max_abs", "variance", "mean_abs"]] = None,
) -> np.ndarray:
    """
    Process SHAP values into a consistent format.

    This function handles multiclass SHAP values by either selecting a specific class
    or aggregating values across classes using one of the supported methods.

    Args:
        shap_explanation (Explanation):
            SHAP explanation object from the explainer.

        class_selection (Optional[Any], optional):
            For multiclass models only: class name or index to select SHAP values for.
            This extracts values for a single specific class instead of aggregating
            across classes. Ignored for binary classification or regression models.
            Default is None (no specific class selected).

        weight_type (Optional[Union[Literal["frequency"], Dict[Any, float]]], optional):
            Determines how to weight SHAP values across classes in multiclass scenarios:

            - 'frequency': Uses equal weights for all classes (1/n_classes). This is useful
              when all classes are equally important regardless of their frequency in the data.

            - dict: Dictionary with class names/indices as keys and weights as values for
              custom weighting. This allows for precise control over the importance of each class.

            The weighting is valuable in multiclass scenarios:
                - When certain classes are more critical to predict correctly, you can
                  emphasize their importance.
                - When classes are imbalanced, you can balance their influence on the overall
                  feature importance.
                - Example: For a 5-class problem with imbalanced classes, you might use
                  `{0: 0.1, 1: 0.3, 2: 0.1, 3: 0.4, 4: 0.1}` to prioritize classes 1 and 3.

            Default is None (no weighting). Note: This parameter is only applicable for
            multiclass models and has no effect on binary classification.

        multiclass_aggregation (Optional[Literal["max_abs", "variance", "mean_abs"]], optional):
            Method to aggregate SHAP values across classes for multiclass models:

            - "max_abs": Maximum absolute SHAP value across classes for each feature.
              This highlights features that are strongly influential for at least one class,
              which is useful for identifying the most discriminative features regardless
              of which class they affect most.

            - "variance": Variance of SHAP values across classes.
              This identifies features whose impact varies significantly between classes,
              which can help detect features that play different roles in differentiating
              various classes.

            - "mean_abs": Mean absolute SHAP value across classes.
              This provides the most balanced measure of feature importance by averaging
              the absolute contribution across all classes, preventing positive and negative
              contributions from canceling each other out.

            Default is None (uses first class for multiclass).

    Returns:
        np.ndarray:
            Processed SHAP values in a consistent format.
    """
    # Determine if the SHAP values are for a multiclass problem (3D) or binary/regression (2D)
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
                " This parameter only applies to multiclass models.",
                UserWarning,
            )
        if weight_type is not None:
            warnings.warn(
                "`weight_type` parameter is ignored for binary classification or regression models."
                " This parameter only applies to multiclass models.",
                UserWarning,
            )
        if multiclass_aggregation is not None:
            warnings.warn(
                "`multiclass_aggregation` parameter is ignored for binary classification or regression models."
                " This parameter only applies to multiclass models.",
                UserWarning,
            )
        return shap_explanation.values

    # For multiclass, get model classes information
    model_classes = _get_model_classes(shap_explanation)

    # Get a copy of the original SHAP values to modify
    shap_values = shap_explanation.values.copy()

    # Handle class selection - highest priority, returns immediately
    if class_selection is not None:
        return _get_shap_values_for_class(shap_explanation, class_selection, model_classes)

    # Apply class weighting if specified (modifies the original 3D values in place)
    if weight_type is not None:
        shap_values = _apply_class_weighting(shap_values, weight_type, model_classes)

    # Apply aggregation across classes if specified (reduces dimension to 2D)
    if multiclass_aggregation is not None:
        return _aggregate_multiclass_shap(shap_values, multiclass_aggregation)

    # If aggregation wasn't applied but we need to reduce dimensions:
    # - If weighting was applied, reduce by summing across classes
    # - If no weighting was applied, return the first class
    if weight_type is not None:
        # Sum weighted values across classes to get final 2D result
        return np.sum(shap_values, axis=2)
    else:
        # Default: return values for the first class (multi-class)
        return shap_values[:, :, 0]


def _get_model_classes(shap_explanation: Explanation) -> List:
    """
    Extract class names/indices from a SHAP explanation object.

    Args:
        shap_explanation (Explanation): SHAP explanation object

    Returns:
        List: List of class names or indices
    """
    if hasattr(shap_explanation, "output_names"):
        return shap_explanation.output_names
    else:
        # If no output_names attribute, use indices based on the third dimension size
        return list(range(shap_explanation.values.shape[2]))


def _get_shap_values_for_class(shap_explanation: Explanation, class_selection: Any, model_classes: List) -> np.ndarray:
    """
    Extract SHAP values for a specific class.

    Args:
        shap_explanation (Explanation): SHAP explanation object
        class_selection: Class name or index to select
        model_classes (List): List of available class names/indices

    Returns:
        np.ndarray: SHAP values for the specified class

    Raises:
        ValueError: If the requested class is not found
    """
    # Validate that the requested class exists
    if class_selection not in model_classes:
        raise ValueError(f"Class '{class_selection}' not found in model classes: {model_classes}")

    # Return SHAP values for the specified class
    class_idx = model_classes.index(class_selection)
    return shap_explanation.values[:, :, class_idx]


def _apply_class_weighting(
    shap_values: np.ndarray, weight_type: Union[Literal["frequency"], Dict[str, float]], model_classes: List
) -> np.ndarray:
    """
    Apply weighting to multiclass SHAP values while preserving the 3D structure.

    Args:
        shap_values (np.ndarray): Original 3D SHAP values
        weight_type: Type of weighting to apply ('frequency' or custom dictionary)
        model_classes (List): List of available class names/indices

    Returns:
        np.ndarray: Weighted SHAP values (still in 3D format)

    Raises:
        ValueError: If an unsupported weight_type is provided
    """
    # Calculate weights for each class
    if weight_type == "frequency":
        # Equal weights for all classes (1/n_classes)
        weights = np.ones(len(model_classes)) / len(model_classes)
    elif isinstance(weight_type, dict):
        # Use user-provided weights from dictionary
        weights = np.array([weight_type.get(cls, 0.0) for cls in model_classes])

        # Normalize weights to sum to 1 if any are non-zero
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are 0, use equal weighting
            weights = np.ones(len(model_classes)) / len(model_classes)
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}. Use 'frequency' or a dictionary of weights.")

    # Apply weights to each class in the 3D array (n_samples, n_features, n_classes)
    # Create weighted values array with the same shape as input
    weighted_values = np.zeros_like(shap_values)

    # Apply weights to each class
    for i, weight in enumerate(weights):
        weighted_values[:, :, i] = shap_values[:, :, i] * weight

    return weighted_values


def _aggregate_multiclass_shap(
    shap_values: np.ndarray, aggregation_method: Literal["max_abs", "variance", "mean_abs"]
) -> np.ndarray:
    """
    Aggregate SHAP values across classes using the specified method.

    Args:
        shap_values (np.ndarray): 3D SHAP values to aggregate (n_samples, n_features, n_classes)
        aggregation_method: Method to use for aggregation

    Returns:
        np.ndarray: Aggregated SHAP values (2D: n_samples, n_features)
    """
    if aggregation_method == "max_abs":
        # Maximum absolute SHAP value across classes for each feature
        return np.max(np.abs(shap_values), axis=2)
    elif aggregation_method == "variance":
        # Variance of SHAP values across classes
        return np.var(shap_values, axis=2)
    elif aggregation_method == "mean_abs":
        # Mean absolute SHAP value across classes
        return np.mean(np.abs(shap_values), axis=2)

    # Should never reach here based on type hinting
    return shap_values[:, :, 0]


def _shap_values_to_df(
    model: BaseEstimator,
    X: pd.DataFrame,
    precalc_shap: Optional[np.ndarray] = None,
    class_selection: Optional[Any] = None,
    multiclass_aggregation: Optional[Literal["max_abs", "variance", "mean_abs"]] = None,
    weight_type: Optional[Union[Literal["frequency"], Dict[Any, float]]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convert SHAP values to a pandas DataFrame with the same structure as the input data.

    This function either uses precalculated SHAP values or calculates them using the
    provided model and dataset, then returns them in a DataFrame format that preserves
    the original feature names and index.

    Args:
        model (BaseEstimator):
            Pretrained model to explain. Used only if precalc_shap is None.

        X (pd.DataFrame):
            Dataset on which the SHAP importance is calculated or was calculated.
            Used to get column names and index for the output DataFrame.

        precalc_shap (np.ndarray, optional):
            Precalculated SHAP values. If None, they are computed using the shap_calc function.
            Default is None.

        class_selection (Optional[Any], optional):
            For multiclass models only: class name or index to select SHAP values for.
            This extracts values for a single specific class instead of aggregating
            across classes. Ignored for binary classification or regression models.
            Default is None (no specific class selected).

        multiclass_aggregation (Optional[Literal["max_abs", "variance", "mean_abs"]], optional):
            Method to aggregate SHAP values across classes for multiclass models:

            - "max_abs": Maximum absolute SHAP value across classes for each feature.
              This highlights features that are strongly influential for at least one class,
              which is useful for identifying the most discriminative features regardless
              of which class they affect most.

            - "variance": Variance of SHAP values across classes.
              This identifies features whose impact varies significantly between classes,
              which can help detect features that play different roles in differentiating
              various classes.

            - "mean_abs": Mean absolute SHAP value across classes.
              This provides the most balanced measure of feature importance by averaging
              the absolute contribution across all classes, preventing positive and negative
              contributions from canceling each other out.

            Default is None (uses first class for multiclass).

        weight_type (Optional[Union[Literal["frequency"], Dict[Any, float]]], optional):
            Determines how to weight SHAP values across classes in multiclass scenarios:

            - 'frequency': Uses equal weights for all classes (1/n_classes). This is useful
              when all classes are equally important regardless of their frequency in the data.

            - dict: Dictionary with class names/indices as keys and weights as values for
              custom weighting. This allows for precise control over the importance of each class.

            The weighting is valuable in multiclass scenarios:
                - When certain classes are more critical to predict correctly, you can
                  emphasize their importance.
                - When classes are imbalanced, you can balance their influence on the overall
                  feature importance.
                - Example: For a 5-class problem with imbalanced classes, you might use
                  `{0: 0.1, 1: 0.3, 2: 0.1, 3: 0.4, 4: 0.1}` to prioritize classes 1 and 3.

            Default is None (no weighting). Note: This parameter is only applicable for
            multiclass models and has no effect on binary classification.

        **shap_kwargs (Any):
                Additional arguments passed to:
                1. SHAP Explainer - parameters like 'approximate' and 'check_additivity'
                2. SHAP values multi-classification conversion - parameters like 'class_selection', 'multiclass_aggregation', and 'weight_type'

                The conversion parameters are extracted internally and control how SHAP values are processed
                for multiclass models.

    Returns:
        pd.DataFrame:
            DataFrame containing SHAP values with the same columns and index as X
            (if X is a DataFrame) or with generated column names (if X is a numpy array).

    Raises:
        ValueError: If input X is empty (has 0 rows or 0 columns).
        TypeError: If X is not a DataFrame or 2D array-like object.
    """
    # Check for empty dataset to avoid meaningless calculations
    if hasattr(X, "shape") and (X.shape[0] == 0 or X.shape[1] == 0):
        raise ValueError("Input X cannot be empty")

    # Use precalculated SHAP values if provided, otherwise calculate them
    if precalc_shap is not None:
        shap_values = precalc_shap
    else:
        shap_explanation = calculate_shap_explanation(model, X, return_explainer=False, **kwargs)
        shap_values = shap_explanation_to_shap_df(
            shap_explanation=shap_explanation,
            model=model,
            X=X,
            class_selection=class_selection,
            multiclass_aggregation=multiclass_aggregation,
            weight_type=weight_type,
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


def extract_shap_multiclass_params(shap_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract parameters related to multiclass SHAP value conversions from shap_kwargs.

    This helper function separates parameters that are passed to the SHAP explainer
    from parameters that control multiclass SHAP value conversion.

    Args:
        shap_kwargs (Dict[str, Any]):
            Dictionary of keyword arguments for SHAP and multiclass processing.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - First dict: Parameters for multiclass SHAP values conversion
            - Second dict: Parameters for SHAP explainer
    """
    # Parameters that are used only for multiclass SHAP value conversion
    multiclass_params = {
        "class_selection": None,
        "multiclass_aggregation": None,
        "weight_type": None,
    }

    # Extract parameters related to multiclass conversion
    extracted_params = {}
    filtered_kwargs = shap_kwargs.copy()

    for param_name in multiclass_params:
        if param_name in filtered_kwargs:
            extracted_params[param_name] = filtered_kwargs.pop(param_name)

    return extracted_params, filtered_kwargs


def prep_shap_related_variables(
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
    _, filtered_shap_kwargs = extract_shap_multiclass_params(shap_kwargs)

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
