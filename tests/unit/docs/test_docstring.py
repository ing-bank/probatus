# This approach is adapted from, and explained in: https://calmcode.io/docs/epic.html

import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pytest

import probatus.features
import probatus.model_interpretation
import probatus.data_comparison
import probatus.metrics

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")

CLASSES_TO_TEST = [
    probatus.features.ShapRFECV,
    probatus.model_interpretation.DependencePlotter,
    probatus.data_comparison.SHAPImportanceResemblance,
    probatus.data_comparison.PermutationImportanceResemblance,
    probatus.metrics.Scorer,
    probatus.model_interpretation.ShapModelInterpreter,
]

FUNCTIONS_TO_TEST: List = []


def handle_docstring(doc, indent):
    """
    Check python code in docstring.

    This function will read through the docstring and grab
    the first python code block. It will try to execute it.
    If it fails, the calling test should raise a flag.
    """
    if not doc:
        return
    start = doc.find("```python\n")
    end = doc.find("```\n")
    if start != -1:
        if end != -1:
            code_part = doc[(start + 10) : end].replace(" " * indent, "")
            exec(code_part)


@pytest.mark.parametrize("c", CLASSES_TO_TEST)
def test_class_docstrings(c):
    """
    Take the docstring of a given class.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(c.__doc__, indent=4)


@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
def test_class_docstrings_lgbm():
    """
    Take the docstring of a given class which uses LightGBM.
    We test that the docstring can be run without errors since it will import LightGBM.

    The test is skipped if the environment does not support LightGBM correctly, such as macos.
    """
    # Test ShapRFECV with early stopping parameters
    handle_docstring(probatus.features.ShapRFECV.__doc__, indent=4)


@pytest.mark.parametrize("f", FUNCTIONS_TO_TEST)
def test_function_docstrings(f):
    """
    Take the docstring of every function.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(f.__doc__, indent=4)
