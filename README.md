# Probatus

## Overview

Developing a well-performing model is often the easy part. What comes next, validating the model, is often overlooked. **Probatus** is a Python library that standardizes and collects different validation steps to be performed on the models. It focuses on binary classification models to test and measure model robustness. 

## Installation

#### Dependencies

Probatus requires:

- scikit-learn (>= 0.20.2)
- pandas (>= 0.25)
- matplotlib (>= 3.1.1)
- seaborn (>= 0.9.0)
- shap (>= 0.32)
- scipy (>= 1.4.0)
- joblib (>= 0.13.2)

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

## Documentation

Latest documentation can be found [here](https://probatus.readthedocs.io/en/latest/).

## Contribution

To learn more about making a contribution to probatus, please see [`CONTRIBUTING.md`](CONTRIBUTING.md).