import setuptools
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="probatus",
    version="1.4.4",
    description="Tools for machine learning model validation",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="ING Bank N.V.",
    author_email="ml_risk_and_pricing_aa@ing.com",
    license="MIT License",
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scikit-learn>=0.22.2",
        "pandas>=0.25",
        "matplotlib>=3.1.1",
        "scipy>=1.4.0",
        "joblib>=0.13.2",
        "tqdm>=4.41.0",
        "shap>=0.36.0"
    ],
    url="https://gitlab.com/ing_rpaa/probatus",
    zip_safe=False,
)
