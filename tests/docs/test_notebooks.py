import pytest
import os

# Turn off interactive mode in plots

plots_disable = "import matplotlib \n" \
                "import matplotlib.pyplot as plt \n" \
                "plt.ioff() \n" \
                "matplotlib.use('Agg') \n" \

NOTEBOOKS_PATH = './docs/tutorials/'
NOTEBOOKS_TO_TEST = [
    'nb_binning',
    'nb_custom_scoring',
    'nb_distribution_statistics',
    'nb_metric_volatility',
    'nb_sample_similarity',
    'nb_shap_feature_elimination',
    'nb_shap_model_interpreter'
]

def get_notebook_path(notebook_name):
    return os.path.join(NOTEBOOKS_PATH, notebook_name + '.ipynb')

@pytest.mark.parametrize("notebook_name", NOTEBOOKS_TO_TEST)
def test_jupyter_notebook(notebook_name):
    notebook_path = get_notebook_path(notebook_name)
    code_to_execute = os.popen(f"jupyter nbconvert --to script --execute --stdout {notebook_path}").read()
    _ = os.popen(f"python3 -c {plots_disable + code_to_execute}").read()