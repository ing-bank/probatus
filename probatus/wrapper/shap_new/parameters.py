import warnings
from typing import Any, Dict, Literal, Tuple


# TODO: Replace by the new one in common
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

    default_shap_tree_explanation_params = {
        "check_additivity": False,
        "approximate": False,
    }

    # Extract parameters related to multi-class conversion
    multi_class_params = {}
    shap_tree_explanation_params = {}
    shap_explainer_params = shap_kwargs.copy()

    # Multi-class parameters
    for param_name in default_multi_class_params:
        if param_name in shap_explainer_params:
            multi_class_params[param_name] = shap_explainer_params.pop(param_name)

    # SHAP explanation parameters
    for param_name in default_shap_tree_explanation_params:
        if param_name in shap_explainer_params:
            shap_tree_explanation_params[param_name] = shap_explainer_params.pop(param_name)

    if verbose > 0:
        if multi_class_params.keys() == default_multi_class_params.keys():
            warnings.warn(
                "No multi-class parameters provided. Default values passed to the SHAP explainer.",
                UserWarning,
            )

    return multi_class_params, shap_explainer_params, shap_tree_explanation_params
