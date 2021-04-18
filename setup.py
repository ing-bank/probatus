import setuptools
import os


def read(fname):
    """
    Read contents of file as a string.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


base_packages = [
    "scikit-learn>=0.22.2",
    "pandas>=1.0.0",
    "matplotlib>=3.1.1",
    "scipy>=1.4.0",
    "joblib>=0.13.2",
    "tqdm>=4.41.0",
    "shap>=0.38.1",
    "numpy>=1.19.0",
]

extra_dep = [
    "scipy>=1.4.0",
]

dev_dep = [
    "flake8>=3.8.3",
    "black>=19.10b0",
    "pre-commit>=2.5.0",
    "mypy>=0.770",
    "flake8-docstrings>=1.4.0",
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
    "pyflakes",
    "seaborn>=0.9.0",
    "joblib>=0.13.2",
    "lightgbm>=3.1.0",
    "jupyter>=1.0.0",
    "tabulate>=0.8.7",
    "nbconvert>=6.0.7",
    "pre-commit>=2.7.1",
]

docs_dep = [
    "mkdocs-material>=6.1.0",
    "mkdocs-git-revision-date-localized-plugin>=0.7.2",
    "mkdocs-git-authors-plugin>=0.3.2",
    "mkdocs-table-reader-plugin>=0.4.1",
    "mkdocs-enumerate-headings-plugin>=0.4.3",
    "mkdocs-awesome-pages-plugin>=2.4.0",
    "mkdocs-minify-plugin>=0.3.0",
    "mknotebooks>=0.6.2",
    "mkdocstrings>=0.13.6",
    "mkdocs-print-site-plugin>=0.8.2",
    "mkdocs-markdownextradata-plugin>=0.1.9",
]

setuptools.setup(
    name="probatus",
    version="1.8.1",
    description="Validation of binary classifiers and data used to develop them",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="ING Bank N.V.",
    author_email="mateusz.garbacz@ing.com",
    license="MIT License",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=base_packages,
    extras_require={
        "extras": base_packages + extra_dep,
        "all": base_packages + extra_dep + dev_dep + docs_dep,
    },
    url="https://github.com/ing-bank/probatus",
    zip_safe=False,
)
