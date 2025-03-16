from typing import Union, List, Any
from sklearn.base import is_regressor, RegressorMixin


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
