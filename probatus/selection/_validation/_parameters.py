from warnings import warn
import numpy as np
from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV


def _validate_min_features_parameter(min_features: int) -> int:
    """
    Validates the min_features_to_select parameter.

    Args:
        min_features (int):
            The minimum number of features to select.

    Returns:
        int:
            The validated min_features value.

    Raises:
        ValueError:
            If min_features is not a positive integer.
    """
    if not isinstance(min_features, (int, np.int64)):
        raise TypeError(f"min_features_to_select must be an integer; got {type(min_features)}.")
    if min_features <= 0:
        raise ValueError(f"min_features_to_select must be > 0; got {min_features}.")
    return min_features


def _validate_model_compatibility_with_early_stopping_parameter(model: Union[BaseEstimator, BaseSearchCV]) -> bool:
    """
    Checks whether the given model supports early stopping.

    If the model is a hyperparameter search object (e.g., `GridSearchCV`, `RandomizedSearchCV`),
    this method checks its base estimator.

    Args:
        model (Union[BaseEstimator, BaseSearchCV]):
            The model or hyperparameter search object to check for early stopping compatibility.

    Returns:
        bool:
            `True` if the model supports early stopping, `False` otherwise.
    """
    # List of supported libraries and their model class names
    # TODO: Add CatBoostClassifier (when it supports NumPy 2.0)
    libraries = [("lightgbm", "LGBMModel"), ("xgboost", "XGBModel")]  # , ("catboost", "CatBoost")]

    # If model is a search CV, get the underlying estimator
    if isinstance(model, BaseSearchCV):
        model = model.estimator

    # Check if model is an instance of any supported class
    for lib, class_name in libraries:
        try:
            module = __import__(lib, fromlist=[class_name])
            model_class = getattr(module, class_name)
            if isinstance(model, model_class):
                return True
        except (ImportError, AttributeError):
            # Skip if library is not installed or class doesn't exist
            warn(f"Library {lib} is not installed or class {class_name} does not exist.")

    return False


def _validate_shap_variance_penalty_factor_parameter(
    shap_variance_penalty_factor: Optional[Union[int, float]],
) -> float:
    """
    Validates the shap_variance_penalty_factor parameter.

    Args:
        shap_variance_penalty_factor (Optional[Union[int, float]]):
            The penalty factor to apply to SHAP values with high variance.

    Returns:
        float:
            The validated shap_variance_penalty_factor value.
    """
    if (
        isinstance(shap_variance_penalty_factor, (int, np.int64, float, np.float64))
        and shap_variance_penalty_factor >= 0
    ):
        return float(shap_variance_penalty_factor)
    else:
        if shap_variance_penalty_factor is not None:
            warn("shap_variance_penalty_factor must be None, int or float. Setting shap_variance_penalty_factor = 0")
        return 0.0


def _validate_step_parameter(step: Union[int, float]) -> Union[int, float]:
    """
    Validates the `step` parameter used for feature elimination.

    The step value determines how many features are removed per iteration.
    It must be a positive integer (absolute count) or a positive float
    (fraction of remaining features).

    Args:
        step (Union[int, float]):
            The step value to validate.

    Returns:
        Union[int, float]:
            The validated step value.

    Raises:
        ValueError:
            If `step` is not a positive integer or float.
    """
    if not isinstance(step, (int, np.int64, float, np.float64)):
        raise TypeError(f"step must be an integer; got {type(step)}.")
    if step <= 0:
        raise ValueError(f"min_features_to_select must be > 0; got {step}.")
    return step
