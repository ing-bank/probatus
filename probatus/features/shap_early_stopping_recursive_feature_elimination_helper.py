from typing import Dict, Any, Optional, Union, Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV


def _get_fit_params_lightGBM(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    train_index: Optional[np.ndarray] = None,
    val_index: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
    eval_metric: Optional[str] = None,
    verbose: Literal[0, 1, 2] = 0,
) -> Dict[str, Any]:
    """
    Prepares the fit parameters for LightGBM models with early stopping support.

    This method formats the necessary parameters for training LightGBM, including
    validation data and optional sample weights.

    Args:
        X_val (pd.DataFrame):
            Validation feature matrix of shape `(n_samples, n_features)`.

        y_val (pd.Series):
            Validation labels of shape `(n_samples,)`.

        sample_weight (Optional[pd.Series], optional):
            Sample weights for training data, if applicable. Default is `None`.

        train_index (Optional[np.ndarray], optional):
            Indices of training samples. Default is `None`.

        val_index (Optional[np.ndarray], optional):
            Indices of validation samples. Default is `None`.

        early_stopping_rounds (Optional[int], optional):
            Number of early stopping rounds. Default is `None`.

        eval_metric (Optional[str], optional):
            Evaluation metric. Default is `None`.

        verbose (int, optional):
            Verbosity level. Default is `0`.

    Returns:
        Dict[str, Any]:
            A dictionary containing the formatted parameters to be passed to
            the LightGBM `fit` method, including validation sets and callbacks.
    """
    from lightgbm import early_stopping

    # Ensure early_stopping_rounds is not None
    if early_stopping_rounds is None:
        raise ValueError("early_stopping_rounds must be provided for LightGBM early stopping")

    # Create the fit parameters with eval_set and callbacks
    fit_params = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": eval_metric,
        "callbacks": [
            early_stopping(early_stopping_rounds, first_metric_only=True, verbose=True if verbose > 1 else False),
        ],
    }

    # Add sample weights if provided
    if sample_weight is not None and train_index is not None:
        fit_params["sample_weight"] = sample_weight.iloc[train_index]

        # Add validation sample weights if validation indices are provided
        if val_index is not None:
            fit_params["eval_sample_weight"] = [sample_weight.iloc[val_index]]

    return fit_params


def _get_fit_params_XGBoost(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    train_index: Optional[np.ndarray] = None,
    val_index: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Prepares the fit parameters for training an XGBoost model with early stopping support.

    This method formats the required parameters for XGBoost training, ensuring the
    correct structure for validation data and sample weights. The early stopping
    configuration is handled separately via the model's parameters.

    Args:
        X_val (pd.DataFrame):
            Validation feature matrix of shape `(n_samples, n_features)`.

        y_val (pd.Series):
            Validation labels of shape `(n_samples,)`.

        sample_weight (Optional[pd.Series], optional):
            Sample weights for training data, if applicable.
            Default is `None`.

        train_index (Optional[np.ndarray], optional):
            Indices of training samples.
            Default is `None`.

        val_index (Optional[np.ndarray], optional):
            Indices of validation samples.
            Default is `None`.

    Returns:
        Dict[str, Any]:
            A dictionary containing the formatted parameters to be passed to the
            XGBoost `fit` method, including validation data.
    """
    # Create fit parameters dictionary with eval_set for validation
    fit_params = {"eval_set": [(X_val, y_val)]}

    # Add sample weights if provided
    if sample_weight is not None and train_index is not None:
        fit_params["sample_weight"] = sample_weight.iloc[train_index]

        # Add validation sample weights if validation indices are provided
        if val_index is not None:
            fit_params["sample_weight_eval_set"] = [sample_weight.iloc[val_index]]

    return fit_params


def _get_fit_params_CatBoost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    train_index: Optional[np.ndarray] = None,
    val_index: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Prepares the fit parameters for training a CatBoost model with early stopping support.

    This method structures the necessary parameters for CatBoost training, ensuring the correct
    format for validation data, optional sample weights, and early stopping.

    Args:
        X_train (pd.DataFrame):
            Training feature matrix of shape `(n_samples, n_features)`.

        X_val (pd.DataFrame):
            Validation feature matrix of shape `(n_samples, n_features)`.

        y_val (pd.Series):
            Validation labels of shape `(n_samples,)`.

        sample_weight (Optional[pd.Series], optional):
            Sample weights for training data, if applicable.
            Default is `None`.

        train_index (Optional[np.ndarray], optional):
            Indices of training samples.
            Default is `None`.

        val_index (Optional[np.ndarray], optional):
            Indices of validation samples.
            Default is `None`.

    Returns:
        Dict[str, Any]:
            A dictionary containing the formatted parameters to be passed to the
            CatBoost `fit` method, including validation data and early stopping settings.
    """
    from catboost import Pool

    # Identify categorical features
    cat_features = [col for col in X_train.select_dtypes(include=["category"]).columns]

    # Create validation data pool
    eval_set_pool = Pool(X_val, y_val, cat_features=cat_features)

    # Add validation sample weights if provided
    if sample_weight is not None and val_index is not None:
        eval_set_pool.set_weight(sample_weight.iloc[val_index])

    # Create fit parameters dictionary with eval_set
    fit_params = {"eval_set": eval_set_pool, "cat_features": cat_features}

    # Add sample weights if provided
    if sample_weight is not None and train_index is not None:
        fit_params["sample_weight"] = sample_weight.iloc[train_index]

    return fit_params


def get_fit_params(
    model: Union[BaseEstimator, BaseSearchCV],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    train_index: Optional[np.ndarray] = None,
    val_index: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
    eval_metric: Optional[str] = None,
    verbose: Literal[0, 1, 2] = 0,
) -> Dict[str, Any]:
    """
    Generates the appropriate fit parameters based on the model type.

    This method automatically detects the model type and delegates to the corresponding
    specialized method (`_get_fit_params_XGBoost`, `_get_fit_params_CatBoost`,
    `_get_fit_params_lightGBM`) to retrieve the correct fit parameters.

    Args:
        model (Union[BaseEstimator, BaseSearchCV]):
            The model or hyperparameter search object for which to retrieve fit parameters.

        X_train (pd.DataFrame):
            Training feature matrix of shape `(n_samples, n_features)`.

        y_train (pd.Series):
            Training labels of shape `(n_samples,)`.

        X_val (pd.DataFrame):
            Validation feature matrix of shape `(n_samples, n_features)`.

        y_val (pd.Series):
            Validation labels of shape `(n_samples,)`.

        sample_weight (Optional[pd.Series], optional):
            Sample weights for training data, if applicable.
            Default is `None`.

        train_index (Optional[np.ndarray], optional):
            Indices of training samples.
            Default is `None`.

        val_index (Optional[np.ndarray], optional):
            Indices of validation samples.
            Default is `None`.

        early_stopping_rounds (Optional[int], optional):
            Number of early stopping rounds. Default is `None`.

        eval_metric (Optional[str], optional):
            Evaluation metric. Default is `None`.

        verbose (int, optional):
            Verbosity level. Default is `0`.

    Returns:
        Dict[str, Any]:
            A dictionary of parameters to be passed to the model's `fit` method,
            including validation data and early stopping settings.

    Raises:
        ValueError:
            If the model type is not supported for early stopping.
    """
    # Try LightGBM
    try:
        from lightgbm import LGBMModel

        if isinstance(model, LGBMModel):
            return _get_fit_params_lightGBM(
                X_val=X_val,
                y_val=y_val,
                sample_weight=sample_weight,
                train_index=train_index,
                val_index=val_index,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
                verbose=verbose,
            )
    except ImportError:
        pass

    # Try XGBoost
    try:
        from xgboost import XGBModel

        if isinstance(model, XGBModel):
            return _get_fit_params_XGBoost(
                X_val=X_val,
                y_val=y_val,
                sample_weight=sample_weight,
                train_index=train_index,
                val_index=val_index,
            )
    except ImportError:
        pass

    # TODO: Revert this once CatBoost is updated to work with NumPy 2.0
    try:
        # Only attempt to import if the model's class name suggests it might be a CatBoost model
        if hasattr(model, "__class__") and "catboost" in str(model.__class__).lower():
            try:
                from catboost import CatBoost

                if isinstance(model, CatBoost):
                    return _get_fit_params_CatBoost(
                        X_train=X_train,
                        X_val=X_val,
                        y_val=y_val,
                        sample_weight=sample_weight,
                        train_index=train_index,
                        val_index=val_index,
                    )
            except ImportError:
                pass
    except Exception:
        # Ignore any errors during this check
        pass

    raise ValueError("Model type not supported for early stopping")
