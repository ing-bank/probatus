def assure_list_of_strings(variable, variable_name: str) -> list[str]:
    """
    Ensure that a variable is a list of strings.

    This utility function converts a single string to a list containing that string,
    keeps a list unchanged, or raises an error for other data types.

    Parameters
    ----------
    variable : str or list
        The variable to check and potentially convert
    variable_name : str
        Name of the variable (used in error message)

    Returns
    -------
    list[str]
        A list containing strings

    Raises
    ------
    ValueError
        If the variable is neither a string nor a list
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
