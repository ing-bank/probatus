<img src="img/logo_large.png" width=190 align="right">

**Probatus** is a Python library that allows to analyse binary classification models as well as the data used to develop them.
The main features assess the metric stability and analyse differences between two data samples e.g. shift between train and test splits.

## Overview

Probatus provides a comprehensive suite of tools for machine learning model analysis and data quality assessment. The library is particularly useful for:

- Model Interpretation
  - SHAP-based feature importance analysis
  - Model behavior visualization

- Feature Selection
  - Recursive feature elimination using SHAP importance
  - Early stopping support for gradient boosted models
  - Cross-validation at each elimination step

- Data Quality Assessment
  - Distribution shift detection
  - Feature importance comparison
  - Dataset similarity analysis

## Key Features

### Model Interpretation
- Comprehensive SHAP analysis for model interpretability
- Support for both tree-based and linear models
- Multiple visualization options for feature importance
- Performance metric stability assessment

### Feature Selection
- SHAP-based recursive feature elimination
- Integration with hyperparameter optimization
- Early stopping support for gradient boosted models
- Cross-validation at each elimination step

### Data Analysis
- Distribution shift detection between datasets
- Feature importance comparison
- Dataset similarity analysis

## Installation

In order to install Probatus you need to use Python 3.9 or higher.

Install `probatus` via pip with:

```bash
pip install probatus
```

Alternatively you can fork/clone and run:

```bash
git clone https://gitlab.com/ing_rpaa/probatus.git
cd probatus
pip install .
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Licence

Probatus is created under MIT License, see more in [LICENCE file](https://github.com/ing-bank/probatus/blob/main/LICENCE).


