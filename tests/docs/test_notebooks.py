import os
from pathlib import Path

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# TODO Fix the breaking notebooks or make them run faster than timeout, then remove the fixed one from the list
FAILING_NOTEBOOKS = {
    "nb_shap_dependence.ipynb",
    "nb_shap_variance_penalty_and_results_comparison.ipynb",
    "nb_rfecv_vs_shaprfecv.ipynb",
}

TIMEOUT_SECONDS = 180
PATH_NOTEBOOKS = [str(path) for path in Path("docs").glob("*/*.ipynb")]
TEST_NOTEBOOKS = bool(os.environ.get("TEST_NOTEBOOKS"))  # Turn on tests by setting TEST_NOTEBOOKS = 1
SKIP_ALL_NB_TESTS = not TEST_NOTEBOOKS


@pytest.mark.parametrize("notebook_path", PATH_NOTEBOOKS)
@pytest.mark.skipif(SKIP_ALL_NB_TESTS, reason="Skip notebook tests if TEST_NOTEBOOK isn't set")
def test_notebook(notebook_path: str) -> None:
    """Run a notebook and check no exception is raised."""
    if Path(notebook_path).name in FAILING_NOTEBOOKS:
        pytest.skip(f"NEEDS FIXING! - Notebook {notebook_path} is either failing or taking too long to run.")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=TIMEOUT_SECONDS, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(Path(notebook_path).parent)}})
