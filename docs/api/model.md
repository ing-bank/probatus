# Model Interpretation using SHAP

The aim of this module is to provide tools for model interpretation using the [SHAP](https://shap.readthedocs.io/en/latest/) library.
The class below is a convenient wrapper that implements multiple plots for tree-based & linear models.

## Overview

The model interpretation module provides comprehensive tools for understanding and visualizing machine learning models using SHAP (SHapley Additive exPlanations). This module is particularly useful for:

- Analyzing feature importance and their impact on model predictions
- Understanding model behavior through various visualization techniques
- Interpreting both tree-based and linear models
- Generating detailed reports of model interpretability

### Key Components

#### ShapModelInterpreter
- A comprehensive wrapper for SHAP analysis
- Provides multiple visualization options:
  - Feature importance plots
  - SHAP summary plots
  - Dependence plots
  - Feature interaction plots
- Supports both classification and regression models
- Handles multi-class scenarios
- Includes model performance metrics

#### ShapDependencePlotter
- Specialized tool for creating SHAP dependence plots
- Helps visualize how individual features affect model predictions
- Supports interaction analysis between features
- Provides customizable plotting options

### Features
- Automatic handling of different model types
- Support for both binary and multi-class classification
- Integration with scikit-learn's API
- Customizable visualization options
- Comprehensive documentation and examples
- Built-in performance metrics calculation

### Use Cases
- Model debugging and validation
- Feature importance analysis
- Model behavior explanation
- Stakeholder communication
- Model optimization guidance

The module is designed to make model interpretation accessible and comprehensive, providing both high-level insights and detailed analysis capabilities.

## Implementation

::: probatus.model.shap_interpreter
::: probatus.model.shap_dependence_plotter