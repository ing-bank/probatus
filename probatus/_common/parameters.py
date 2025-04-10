from typing import Any, Dict, Tuple


def extract_parameters(
    kwargs: Dict[str, Any],
    default_kwarg_dicts: list[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Extract parameters related to multi-class SHAP value conversions from plot_kwargs.

    This helper function separates parameters that are passed to the SHAP explainer
    from parameters that control multi-class SHAP value conversion.

    Args:
        plot_kwargs (Dict[str, Any]):
            Dictionary of keyword arguments for SHAP and multi-class processing.

        default_aggregation_method (Optional[Literal["mean", "max_abs", "mean_abs"]], optional):
            Default aggregation method to use if not specified in plot_kwargs.
            Default is None (no default aggregation method).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - First dict: Parameters for multi-class SHAP values conversion
            - Second dict: Parameters for SHAP explainer
    """
    dict_list = []

    # Iterate over all dicts
    for default_kwarg_dict in default_kwarg_dicts:
        # filled dicts insertion
        temp_dict = {}
        for param_name in default_kwarg_dict:
            if param_name in kwargs:
                temp_dict[param_name] = kwargs.pop(param_name)
            else:
                temp_dict[param_name] = default_kwarg_dict[param_name]
        dict_list.append(temp_dict)

    return kwargs, *dict_list
