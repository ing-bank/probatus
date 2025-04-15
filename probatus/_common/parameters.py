from typing import Any, Dict, List
import inspect
from typing import get_origin, get_args, Union


def extract_parameters(
    kwargs: Dict[str, Any],
    default_kwarg_dicts: list[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # TODO: Remove this function
    """
    Extract parameters

    Args:
        kwargs (Dict[str, Any]):
            Dictionary of keyword arguments.

        default_kwarg_dicts (List[Dict[str, Any]]):
            List of dictionaries with default parameters.

    Returns:
        List[Dict[str, Any]]:
            List of dictionaries with parameters.
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


def get_valid_kwargs(func, kwargs):
    sig = inspect.signature(func)
    valid_kwargs = {}
    for name, value in kwargs.items():
        if name in sig.parameters and value is not None:
            param = sig.parameters[name]
            if _type_matches(value, param.annotation):
                valid_kwargs[name] = value
            else:
                raise ValueError(f"Invalid type for parameter {name}: {type(value)}")

    return valid_kwargs


def _type_matches(value, annotation):
    # No type defined, so assume valid.
    if annotation is inspect.Parameter.empty:
        return True
    # Handle Union types (including Optional)
    if get_origin(annotation) is Union:
        return any(isinstance(value, arg) for arg in get_args(annotation))
    # Regular type checking.
    return isinstance(value, annotation)
