import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from shap import Explainer
from shap.explainers import TreeExplainer
from shap.utils import sample
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def shap_calc(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    return_explainer: bool = False,
    verbose: int = 0,
    random_state: Optional[int] = None,
    sample_size: int = 100,
    approximate: bool = False,
    check_additivity: bool = True,
    **shap_kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, Explainer]]:
    """
    Calculate the SHAP (SHapley Additive exPlanations) values for a given model.

    SHAP values help explain the output of machine learning models by attributing
    the prediction to each input feature.

    Args:
        model (BaseEstimator):
            Trained model to explain. Should be compatible with SHAP library.

        X (pd.DataFrame or np.ndarray):
            Feature set used to calculate SHAP values. Should have the same format
            as the data used to train the model.

        return_explainer (bool, optional):
            If True, returns a tuple (shap_values, explainer) to allow reuse of the explainer.
            Default is False.

        verbose (int, optional):
            Controls verbosity of the output:
            - 0 - neither prints nor warnings are shown
            - 1 - only most important warnings
            - 2 - shows all prints and all warnings.
            Default is 0.

        random_state (int, optional):
            Random state for reproducibility when sampling background data.
            If None, results may not be reproducible. Default is None.

        sample_size (int, optional):
            Number of samples to use for background data when creating the explainer.
            Default is 100.

        approximate (bool, optional):
            If True, uses SHAP approximations for tree-based models - less accurate, but much faster.
            Only applies to tree-based explainers. Default is False.

        check_additivity (bool, optional):
            If False, SHAP will disable the additivity check for tree-based models,
            which can improve performance. Default is True.

        **shap_kwargs:
            Additional keyword arguments passed to the SHAP Explainer.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, Explainer]]:
            SHAP values for the model, optionally also returns the explainer if return_explainer=True.
            For binary classification, returns SHAP values for the positive class.

    Raises:
        TypeError: If the provided model is a Pipeline, which is not supported.
    """
    # SHAP doesn't work with scikit-learn pipelines, so we check for this first
    if isinstance(model, Pipeline):
        raise TypeError(
            "The provided model is a Pipeline. Unfortunately, the features based on SHAP do not support "
            "pipelines, because they cannot be used in combination with shap.Explainer. Please apply any "
            "data transformations before running the probatus module."
        )

    # Suppress warnings regarding XGboost and Lightgbm models if verbose level is low
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if verbose <= 1 else "default")

        # Check if the dataset has categorical features (only possible with pandas DataFrames)
        # This affects how we create the masker for the SHAP explainer
        has_categorical = False
        if isinstance(X, pd.DataFrame):
            has_categorical = X.select_dtypes("category").shape[1] > 0

        # Create a background dataset (masker) for non-tree models
        # Tree models don't need a background dataset unless feature_perturbation is not tree_path_dependent
        masker = None
        if not has_categorical and shap_kwargs.get("feature_perturbation") != "tree_path_dependent":
            # If the dataset is smaller than the requested sample size, use a percentage of the dataset
            if X.shape[0] < sample_size:
                sample_size = int(np.ceil(X.shape[0] * 0.2))  # Use 20% of the dataset

            # Create background data by sampling from the input dataset
            masker = sample(X, sample_size, random_state=random_state)

        # Initialize the SHAP explainer with the model and masker
        explainer = Explainer(model, masker=masker, seed=random_state, **shap_kwargs)

        # Calculate SHAP values - different methods for tree-based vs other models
        if isinstance(explainer, TreeExplainer):
            # Tree-based models can use approximation for faster calculation
            shap_values = explainer.shap_values(X, check_additivity=check_additivity, approximate=approximate)
        else:
            # Standard calculation for non-tree models
            shap_values = explainer.shap_values(X)

        # Handle different output formats from SHAP
        # SHAP can return values in different formats depending on the model type
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # For binary classification, SHAP often returns a list with values for both classes
            # We want to use the values for the positive class (index 1)
            warnings.warn(
                "Shap values are related to the output probabilities of class 1 for this model, instead of log odds."
            )
            shap_values = shap_values[1]  # Take positive class (index 1)
        elif not isinstance(shap_values, list) and len(shap_values.shape) == 3:
            # For some models, SHAP returns a 3D array with shape (samples, features, classes)
            try:
                # Try to get the positive class (index 1) for binary classification
                shap_values = shap_values[:, :, 1]
            except IndexError:
                # If index 1 doesn't exist (e.g., only one class), use the last dimension
                warnings.warn("Could not extract dimension 1 from 3D SHAP values. Using the last dimension instead.")
                shap_values = shap_values[:, :, -1]

    # Return the SHAP values, and optionally the explainer for reuse
    if return_explainer:
        return shap_values, explainer
    return shap_values


def shap_to_df(
    model: BaseEstimator, X: Union[pd.DataFrame, np.ndarray], precalc_shap: Optional[np.ndarray] = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Convert SHAP values to a pandas DataFrame with the same structure as the input data.

    This function either uses precalculated SHAP values or calculates them using the
    provided model and dataset, then returns them in a DataFrame format that preserves
    the original feature names and index.

    Args:
        model (BaseEstimator):
            Pretrained model to explain. Used only if precalc_shap is None.

        X (pd.DataFrame or np.ndarray):
            Dataset on which the SHAP importance is calculated or was calculated.
            Used to get column names and index for the output DataFrame.

        precalc_shap (np.ndarray, optional):
            Precalculated SHAP values. If None, they are computed using shap_calc.
            Default is None.

        **kwargs:
            Additional keyword arguments passed to the shap_calc function if
            precalc_shap is None.

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
    shap_values = precalc_shap if precalc_shap is not None else shap_calc(model, X, **kwargs)

    # Check if X is a pandas DataFrame to access columns and index
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    elif isinstance(X, np.ndarray) and len(X.shape) == 2:
        # For numpy arrays, create generic column names
        return pd.DataFrame(shap_values, columns=[f"col_{ix}" for ix in range(X.shape[1])])
    # Handle other array-like objects that have a shape attribute
    elif hasattr(X, "shape") and len(X.shape) == 2:
        return pd.DataFrame(shap_values, columns=[f"col_{ix}" for ix in range(X.shape[1])])
    else:
        # If X is neither a DataFrame nor a 2D array-like object, raise an error
        raise TypeError("X must be a dataframe or a 2d array-like object")


def calculate_shap_importance(
    shap_values: np.ndarray,
    columns: List[str],
    output_columns_suffix: str = "",
    shap_variance_penalty_factor: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """
    Calculate feature importance metrics based on SHAP values.

    This function computes the average SHAP value and average absolute SHAP value
    for each feature, optionally applying a variance penalty to favor features
    with more consistent impact.

    Args:
        shap_values (np.ndarray):
            Array of SHAP values with shape (n_samples, n_features) or
            (n_classes, n_samples, n_features) for multi-class problems.

        columns (List[str]):
            List of feature names corresponding to the columns in shap_values.
            Must have the same length as the feature dimension of shap_values.

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
