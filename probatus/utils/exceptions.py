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


class NotFittedError(Exception):
    """
    Error.
    """

    def __init__(self, message):
        """
        Init error.
        """
        self.message = message


class DimensionalityError(Exception):
    """
    Error.
    """

    def __init__(self, message):
        """
        Init error.
        """
        self.message = message


class UnsupportedModelError(Exception):
    """
    Error.
    """

    def __init__(self, message):
        """
        Init error.
        """
        self.message = message


class NotInstalledError:
    """
    Raise error when a dependency is not installed.

    This object is used for optional dependencies.
    This allows us to give a friendly message to the user that they need to install extra dependencies as well as a link
    to our documentation page.

    Adapted from: https://github.com/RasaHQ/whatlies/blob/master/whatlies/error.py

    Example usage:

    ```python
    from probatus.utils import NotInstalledError
    try:
        import dash_core_components as dcc
    except ModuleNotFoundError as e:
        dcc = NotInstalledError("dash_core_components", "dashboard")
    dcc.Markdown() # Will raise friendly error with instructions how to solve
    ```
    Note that installing optional dependencies in a package are defined in setup.py.
    """

    def __init__(self, tool, dep=None):
        """
        Initialize error with missing package and reference to conditional install package.

        Args:
            tool (str): The name of the pypi package that is missing
            dep (str): The name of the extra_imports set (defined in setup.py) where the package is present. (optional)
        """
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        if self.dep is None:
            msg += f"pip install {self.tool}\n\n"
        else:
            msg += f"pip install probatus[{self.dep}]\n\n"

        msg += "See probatus installation guide here: https://ing-bank.github.io/probatus/index.html"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        """Raise when accessing an attribute."""
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        """Raise when accessing a method."""
        raise ModuleNotFoundError(self.msg)
