# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def class_name_from_object(obj):
    """
    Helper to quickly retrieve a class name from an object.
    """
    return obj.__class__.__name__


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


def assure_list_values_allowed(variable, variable_name, allowed_values):
    """
    Assert list.
    """
    for value in variable:
        if value not in allowed_values:
            raise (ValueError("Value {} in variable {} is not allowed").format(value, variable_name))
