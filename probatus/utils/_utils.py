def assure_list_of_strings(variable, variable_name):
    """
    Make sure object is a list of strings.
    """
    if isinstance(variable, list):
        return variable
    elif isinstance(variable, str):
        return [variable]
    else:
        raise (ValueError("{} needs to be either a string or list of strings.").format(variable_name))
