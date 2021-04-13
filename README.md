<img src="https://github.com/ing-bank/probatus/raw/main/docs/img/logo_large.png" width="120" align="right">

[![pytest](https://github.com/ing-bank/probatus/workflows/Development/badge.svg)](https://github.com/ing-bank/probatus/actions?query=workflow%3A%22Development%22)
[![codecov](https://codecov.io/gh/ing-bank/probatus/branch/main/graph/badge.svg?token=OFE2YWHLFK)](https://codecov.io/gh/ing-bank/probatus)
[![PyPi Version](https://img.shields.io/pypi/pyversions/probatus)](#)
[![PyPI](https://img.shields.io/pypi/v/probatus)](#)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/probatus)](#)
![GitHub contributors](https://img.shields.io/github/contributors/ing-bank/probatus)

# Probatus

## Overview

**Probatus** is a python package that helps validate binary classification models and the data used to develop them. Main features:

- [probatus.interpret](https://ing-bank.github.io/probatus/api/model_interpret.html) provides shap-based model interpretation tools 
- [probatus.metric_volatility](https://ing-bank.github.io/probatus/api/metric_volatility.html) provides tools using bootstrapping and/or different random seeds to assess metric volatility/stability.
- [probatus.sample_similarity](https://ing-bank.github.io/probatus/api/sample_similarity.html) to compare two datasets using resemblance modelling, f.e. `train` with out-of-time `test`.
- [probatus.feature_elimination.ShapRFECV](https://ing-bank.github.io/probatus/api/feature_elimination.html) provides cross-validated Recursive Feature Elimination using shap feature importance.
- [probatus.missing_values](https://ing-bank.github.io/probatus/api/imputation_selector.html) compares performance gains of different missing values imputation strategies for a given model.

## Installation

```bash
pip install probatus
```

## Documentation

Documentation at [ing-bank.github.io/probatus/](https://ing-bank.github.io/probatus/).

## Contributing

To learn more about making a contribution to probatus, please see [`CONTRIBUTING.md`](CONTRIBUTING.md).
