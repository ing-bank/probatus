import numpy as np
import pandas as pd
from shap import Explainer, Explanation
from shap.explainers import TreeExplainer
from shap.utils import sample
from sklearn.base import BaseEstimator
import warnings

from probatus._common.data_processing import get_pipeline_estimator_and_preprocessor, preprocess_using_pipeline

from sklearn.pipeline import Pipeline
from typing import Any, Literal, Optional, Tuple, Union


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
    shap_explanation = _compute_shap_explanation(
        explainer=explainer, X=X, approximate=approximate, check_additivity=check_additivity
    )

    # Return the SHAP values, and the explainer for reuse
    if return_explainer:
        return shap_explanation, explainer
    # Return the SHAP explanation and the explainer
    return shap_explanation


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


def _compute_shap_explanation(
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
