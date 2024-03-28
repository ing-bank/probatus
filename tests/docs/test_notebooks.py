import os
from pathlib import Path

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

TIMEOUT_SECONDS = 1800
PATH_NOTEBOOKS = [str(path) for path in Path("docs").glob("*/*.ipynb")]
NB_FLAG = os.environ.get("TEST_NOTEBOOKS")  # Turn on tests by setting TEST_NOTEBOOKS = 1
TEST_NOTEBOOKS = False if NB_FLAG == "1" else True


@pytest.mark.parametrize("notebook_path", PATH_NOTEBOOKS)
@pytest.mark.skipif(TEST_NOTEBOOKS, reason="Skip notebook tests if TEST_NOTEBOOK isn't set")
def test_notebook(notebook_path: str) -> None:
    """Run a notebook and check no exception is raised."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=TIMEOUT_SECONDS, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(Path(notebook_path).parent)}})
