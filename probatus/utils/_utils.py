from typing import Union, List


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
