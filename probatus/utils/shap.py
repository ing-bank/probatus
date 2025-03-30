import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from shap import Explainer, Explanation
from shap.explainers import TreeExplainer
from shap.utils import sample
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from probatus.utils.common import get_pipeline_estimator_and_preprocessor, preprocess_using_pipeline


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

            The SHAP Explanation object contains the following attributes:
            - values: SHAP values for each feature
            - base_values: Base value for the model
            - data: Data used to calculate SHAP values

            This can be dimensions like (n_samples, n_features) for regression/binary
            classification problems, or (n_samples, n_features, n_classes) for multi-class
            classification problems.

    Raises:
        TypeError: If the model is not compatible with SHAP.
    """
    # Check prerequisites
    is_valid, error_message = _validate_shap_inputs(model, X, verbose)
    if not is_valid:
        raise TypeError(error_message)

    if isinstance(model, Pipeline):
        # Get the preprocessor and estimator from the pipeline
        model, preprocessor = get_pipeline_estimator_and_preprocessor(model)

        if preprocessor is not None and verbose > 1:
            warnings.warn("Applying preprocessing steps from pipeline before calculating SHAP values.")

        # Apply preprocessing to X if a preprocessor exists
        X = preprocess_using_pipeline(X, model)

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
    shap_values: np.ndarray = format_shap_values(
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


def calculate_shap_importance(
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
        return _process_multiclass_shap_statistics(shap_values, abs_shap_values, penalty_factor)
    else:
        # Process binary/regression SHAP values
        return _process_binary_shap_statistics(shap_values, abs_shap_values, penalty_factor)


def _process_multiclass_shap_statistics(
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


def _process_binary_shap_statistics(
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


def format_shap_values(
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
        return aggregate_multiclass_shap(shap_values, multiclass_aggregation, shap_variance_penalty_factor)

    # If aggregation wasn't applied but we need to reduce dimensions:
    # - If weighting was applied, reduce by summing across classes
    # - If no weighting was applied, return the first class
    if weights is not None:
        # Sum weighted values across classes to get final 2D result
        return np.sum(shap_values, axis=2)
    else:
        # Default: return all multi-class SHAP values
        return shap_values


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


def aggregate_multiclass_shap(
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


def _shap_values_to_df(
    model: BaseEstimator,
    X: pd.DataFrame,
    precalc_shap: Optional[np.ndarray] = None,
    class_selection: Optional[Any] = None,
    multiclass_aggregation: Optional[Literal["mean", "max_abs", "mean_abs"]] = None,
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
        shap_values = shap_explanation_to_shap_df(
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


def extract_shap_multiclass_params(
    shap_kwargs: Dict[str, Any], default_aggregation_method: Optional[Literal["mean", "max_abs", "mean_abs"]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - First dict: Parameters for multi-class SHAP values conversion
            - Second dict: Parameters for SHAP explainer
    """
    # Parameters that are used only for multi-class SHAP value conversion
    multiclass_params = {
        "class_selection": None,
        "multiclass_aggregation": None,
        "weights": None,
        "shap_variance_penalty_factor": None,
    }

    # If default aggregation method is provided, use it as the default aggregation method
    if default_aggregation_method is not None:
        multiclass_params["multiclass_aggregation"] = default_aggregation_method

    # Extract parameters related to multi-class conversion
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
