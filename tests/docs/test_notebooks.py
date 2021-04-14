# This approach is adapted from, and explained in: https://calmcode.io/docs/epic.html

import pytest
import os

# Turn off interactive mode in plots

plots_disable = "import matplotlib \n" "import matplotlib.pyplot as plt \n" "plt.ioff() \n" "matplotlib.use('Agg') \n"
NOTEBOOKS_PATH = "./docs/tutorials/"

NOTEBOOKS_TO_TEST_LGBM = [
    "./docs/tutorials/nb_shap_feature_elimination",
    "./docs/tutorials/nb_imputation_comparision",
    "./docs/discussion/nb_rfecv_vs_shaprfecv.ipynb",
]

NOTEBOOKS_TO_TEST = [
    "./docs/tutorials/nb_binning",
    "./docs/tutorials/nb_custom_scoring",
    "./docs/tutorials/nb_distribution_statistics",
    "./docs/tutorials/nb_metric_volatility",
    "./docs/tutorials/nb_sample_similarity",
    "./docs/tutorials/nb_shap_model_interpreter" "./docs/howto/reproducibility",
]


def execute_notebook_test(notebook_name):
    """
    Execute a notebook.
    """
    notebook_path = notebook_name + ".ipynb"

    code_to_execute = os.popen(
        f"jupyter nbconvert --to script --execute --stdout --PythonExporter.exclude_markdown=True {notebook_path}"
    ).read()
    _ = os.popen(f"python3 -c {plots_disable + code_to_execute}").read()


@pytest.mark.parametrize("notebook_name", NOTEBOOKS_TO_TEST_LGBM)
@pytest.mark.skipif(os.environ.get("SKIP_LIGHTGBM") == "true", reason="LightGBM tests disabled")
@pytest.mark.skip(reason="GitHub pipelines are failing on these tests. Could not find a solution for now.")
def test_jupyter_notebook_lgbm(notebook_name):
    """
    Test a notebook.
    """
    execute_notebook_test(notebook_name)
