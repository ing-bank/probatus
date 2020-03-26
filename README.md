[![pipeline status](https://gitlab.com/ing_rpaa/probatus/badges/master/pipeline.svg)](https://gitlab.com/ing_rpaa/probatus/-/commits/master)
[![coverage report](https://gitlab.com/ing_rpaa/probatus/badges/master/coverage.svg)](https://gitlab.com/ing_rpaa/probatus/-/commits/master)
[![PyPi Version](https://img.shields.io/pypi/pyversions/probatus)](#)
[![PyPI](https://img.shields.io/pypi/v/probatus)](#)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/probatus)](#)

# Probatus

## Overview

Developing a well-performing model is often the easy part. What comes next, validating the model, is often overlooked. **Probatus** is a Python library that standardizes and collects different validation steps to be performed on the models. It focuses on binary classification models to test and measure model robustness. 

## Installation

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

- scikit-learn (>= 0.20.2)
- pandas (>= 0.25)
- matplotlib (>= 3.1.1)
- scipy (>= 1.4.0)
- joblib (>= 0.13.2)

For packages required to develop, please refer to requirements.txt.

## Documentation

Latest documentation can be found [here](https://probatus.readthedocs.io/en/latest/).

## Contribution

To learn more about making a contribution to probatus, please see [`CONTRIBUTING.md`](CONTRIBUTING.md).