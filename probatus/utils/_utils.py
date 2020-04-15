def class_name_from_object(obj):
    return obj.__class__.__name__

def assure_list_of_strings(variable, variable_name):
    if isinstance(variable, list):
        return variable
    elif isinstance(variable, str):
        return [variable]
    else:
        raise(ValueError('{} needs to be either a string or list of strings.').format(variable_name))


def assure_list_values_allowed(variable, variable_name, allowed_values):
    for value in variable:
        if value not in allowed_values:
            raise(ValueError('Value {} in variable {} is not allowed').format(value, variable_name))
