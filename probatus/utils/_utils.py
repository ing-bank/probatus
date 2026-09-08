from __future__ import annotations

from typing import TypeVar, overload

_Feature = TypeVar("_Feature")


@overload
def assure_list_of_strings(variable: str, variable_name: str) -> list[str]: ...


@overload
def assure_list_of_strings(variable: list[_Feature], variable_name: str) -> list[_Feature]: ...


def assure_list_of_strings(variable: str | list[_Feature], variable_name: str) -> list[str] | list[_Feature]:
    """
    Make sure object is a list of strings.
    """
    if isinstance(variable, list):
        return variable
    elif isinstance(variable, str):
        return [variable]
    else:
        raise (ValueError("{} needs to be either a string or list of strings.").format(variable_name))
