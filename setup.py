import setuptools
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

base_packages = ["scikit-learn>=0.20.2",
                 "pandas>=0.25",
                 "matplotlib>=3.1.1",
                 "seaborn>=0.9.0",
                 "shap>=0.32",
		        "joblib>=0.13.2"]

try:
    if os.environ.get('CI_COMMIT_TAG'):
        version = os.environ['CI_COMMIT_TAG']
    else:
        version = os.environ['CI_JOB_ID']
except:
    version = 'local'


setuptools.setup(
    name='probatus',
    version=version,
    description='Tools for machine learning model validation',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='RPAA ING',
    author_email='ml_risk_and_pricing_aa@ing.com',
    license='ING Open Source',
    packages=setuptools.find_packages(),
    package_data={'probatus': ['datasets/data/*.pkl']},
    install_requires=base_packages,
    url='https://gitlab.com/ing_rpaa/probatus',
    zip_safe=False
)
