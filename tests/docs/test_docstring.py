# This approach is adapted from, and explained in: https://calmcode.io/docs/epic.html

import pytest
import matplotlib.pyplot as plt
import matplotlib
import os

import probatus.binning
import probatus.feature_elimination
import probatus.interpret
import probatus.metric_volatility
import probatus.sample_similarity
import probatus.stat_tests
import probatus.utils
import probatus.missing_values

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")

CLASSES_TO_TEST = [
    probatus.binning.SimpleBucketer,
    probatus.binning.AgglomerativeBucketer,
    probatus.binning.QuantileBucketer,
    probatus.binning.TreeBucketer,
    probatus.feature_elimination.ShapRFECV,
    probatus.interpret.DependencePlotter,
    probatus.interpret.ShapModelInterpreter,
    probatus.metric_volatility.TrainTestVolatility,
    probatus.metric_volatility.BootstrappedVolatility,
    probatus.metric_volatility.SplitSeedVolatility,
    probatus.sample_similarity.SHAPImportanceResemblance,
    probatus.sample_similarity.PermutationImportanceResemblance,
    probatus.stat_tests.DistributionStatistics,
    probatus.stat_tests.AutoDist,
    probatus.utils.Scorer,
    probatus.missing_values.ImputationSelector,
]

CLASSES_TO_TEST_LGBM = [
    probatus.feature_elimination.EarlyStoppingShapRFECV,
]

FUNCTIONS_TO_TEST = [
    probatus.utils.sample_row,
]


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
@pytest.mark.parametrize("c", CLASSES_TO_TEST_LGBM)
def test_class_docstrings_lgbm(c):
    """
    Take the docstring of a given class which uses LightGBM.

    The test passes if the usage examples causes no errors.

    The test is skipped if the environment does not support LightGBM correctly, such as macos.
    """
    handle_docstring(c.__doc__, indent=4)


@pytest.mark.parametrize("f", FUNCTIONS_TO_TEST)
def test_function_docstrings(f):
    """
    Take the docstring of every function.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(f.__doc__, indent=4)
