import setuptools
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="probatus",
    version="1.5.1",
    description="Validation of binary classifiers and data used to develop them",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="ING Bank N.V.",
    author_email="ml_risk_and_pricing_aa@ing.com",
    license="MIT License",
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scikit-learn>=0.22.2",
        "pandas>=1.0.0",
        "matplotlib>=3.1.1",
        "scipy>=1.4.0",
        "joblib>=0.13.2",
        "tqdm>=4.41.0",
        "shap>=0.38.1",
        "numpy>=1.19.0"
    ],
    url="https://github.com/ing-bank/probatus",
    zip_safe=False,
)
