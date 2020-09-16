<img src="docs/source/logo_large_white.png" width="120" align="right">

[![pipeline status](https://gitlab.com/ing_rpaa/probatus/badges/master/pipeline.svg)](https://gitlab.com/ing_rpaa/probatus/-/commits/master)
[![coverage report](https://gitlab.com/ing_rpaa/probatus/badges/master/coverage.svg)](https://gitlab.com/ing_rpaa/probatus/-/commits/master)
[![PyPi Version](https://img.shields.io/pypi/pyversions/probatus)](#)
[![PyPI](https://img.shields.io/pypi/v/probatus)](#)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/probatus)](#)

# Probatus

## Overview

**Probatus** is a Python library that standardizes and collects different validation steps to be performed on the models. It focuses on binary classification models to test and measure model robustness. 

## Installation

In order to install probatus you need to use Python 3.6 or higher.

You can install probatus using pip:

```bash
pip install probatus
```

Alternatively you can fork/clone and run:

```bash
git clone https://gitlab.com/ing_rpaa/probatus.git
cd probatus
pip install .
```

#### Dependencies

Probatus requires to install:

- scikit-learn (>= 0.22.2)
- pandas (>= 0.25)
- matplotlib (>= 3.1.1)
- scipy (>= 1.4.0)
- joblib (>= 0.13.2)
- tqdm (>= 4.41.0)
- shap (>=0.36.0)


For packages required for development, please refer to requirements.txt.

## Documentation

Latest documentation can be found [here](https://probatus.readthedocs.io/en/latest/).

## Contribution

To learn more about making a contribution to probatus, please see [`CONTRIBUTING.md`](CONTRIBUTING.md).