import setuptools
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="probatus",
    version="1.2.0",
    description="Tools for machine learning model validation",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="ING Bank N.V.",
    author_email="ml_risk_and_pricing_aa@ing.com",
    license=(
        "Copyright (c) 2020 ING Bank N.V. \n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n"
        "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
    ),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(
        exclude=["probatus.interpret", "tests.interpret",]
    ),
    install_requires=[
        "scikit-learn>=0.22.2",
        "pandas>=0.25",
        "matplotlib>=3.1.1",
        "scipy>=1.4.0",
        "joblib>=0.13.2",
        "tqdm>=4.41.0",
        "shap>=0.36.0",
    ],
    url="https://gitlab.com/ing_rpaa/probatus",
    zip_safe=False,
)
