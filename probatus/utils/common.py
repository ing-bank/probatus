from typing import Union, List, Any, Optional, Dict
from sklearn.base import is_regressor, RegressorMixin
import pandas as pd


def assure_list_of_strings(variable: Union[str, List[str]], variable_name: str) -> List[str]:
    """
    Ensure that a variable is a list of strings.

    This utility function provides type conversion and validation for string inputs.
    It handles three cases:
    1. If input is already a list of strings, returns it unchanged
    2. If input is a single string, converts it to a single-element list
    3. If input is neither, raises a ValueError

    Args:
        variable (Union[str, List[str]]): The variable to check and potentially convert.
            Can be either:
            - A single string
            - A list of strings
            Any other type will raise an error.

        variable_name (str): Name of the variable (used in error message).
            This helps create more informative error messages by including
            the actual variable name in the message.

    Returns:
        List[str]: A list containing one or more strings.
            - If input was a list, returns the same list
            - If input was a string, returns [string]

    Raises:
        ValueError: If the variable is neither a string nor a list.
            The error message will include the provided variable_name
            to help identify which variable caused the error.
    """
    # Case 1: Already a list - return as is
    if isinstance(variable, list):
        return variable
    # Case 2: Single string - convert to a list with one element
    elif isinstance(variable, str):
        return [variable]
    # Case 3: Neither a list nor a string - raise an error
    else:
        raise ValueError(f"{variable_name} needs to be either a string or list of strings.")


def is_regression_model(model: Any) -> bool:
    """
    Determine if a model is a regression model using scikit-learn's built-in functions.

    This function checks if a model is a regressor by using scikit-learn's is_regressor
    function or by checking if it's an instance of RegressorMixin. It also performs
    additional checks for models that might not directly inherit from scikit-learn's
    base classes.

    Args:
        model (Any): The model to check. Can be any scikit-learn estimator,
            pipeline, or compatible model with similar interface.

    Returns:
        bool: True if the model is a regression model, False otherwise.
    """
    # First try scikit-learn's built-in function
    if is_regressor(model):
        return True

    # Check if it's an instance of RegressorMixin
    if isinstance(model, RegressorMixin):
        return True

    # For models that might be wrapped (e.g., in a Pipeline)
    if hasattr(model, "steps") and len(getattr(model, "steps", [])) > 0:
        # Check the final estimator in a pipeline
        final_estimator = model.steps[-1][1]
        return is_regression_model(final_estimator)

    # For other model types, check common attributes
    # Regression models typically have predict but not predict_proba
    has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
    has_predict_proba = hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))

    # If it has predict but not predict_proba, it's likely a regression model
    if has_predict and not has_predict_proba:
        return True

    return False


def handle_class_names(
    y: pd.Series,
    class_names: Optional[Union[List[str], Dict[Union[int, str], str]]] = None,
    is_regression: bool = False,
) -> List[str]:
    """
    Handle class names for visualization based on the input type.

    Args:
        y (pd.Series):
            Target variable series to process.

        class_names (Optional[Union[List[str], Dict[Union[int, str], str]]], optional):
            Either a list of class names that will be mapped to the sorted unique values in y,
            or a dictionary mapping target values to class names.
            If None, default labels will be used. Default is None.

        is_regression (bool, optional):
            Whether the model is a regression model. Default is False.

    Returns:
        List[str]: The list of class names to use for visualization.

    Raises:
        ValueError: If number of class names doesn't match number of unique target values,
                   or if a target value is not found in the provided dictionary.
        TypeError: If class_names is not None, a list, or a dictionary.
    """
    # For regression, always use a single class name
    if is_regression:
        if class_names is None:
            return ["Regression Output"]
        else:
            return class_names

    # For classification, process class names based on input type
    unique_y_values = sorted(y.unique())
    n_classes = len(unique_y_values)

    if class_names is None:
        # If no class names provided, use default labels
        return [f"label_{i}" for i in range(n_classes)]
    elif isinstance(class_names, list):
        # If list provided, check if it matches the number of classes
        if len(class_names) != n_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) must match number of unique target values ({n_classes})"
            )
        return class_names
    elif isinstance(class_names, dict):
        # If dictionary provided, extract class names based on the mapping
        # Convert all keys to strings for consistent comparison
        class_name_dict = {str(k): v for k, v in class_names.items()}
        # Create list of class names in order of unique values
        result_class_names = []
        for val in unique_y_values:
            if str(val) in class_name_dict:
                result_class_names.append(class_name_dict[str(val)])
            else:
                raise ValueError(f"Target value {val} not found in the class_names dictionary")
        return result_class_names
    else:
        raise TypeError(
            "class_names must be None, a list of strings, or a dictionary mapping target values to class names"
        )
