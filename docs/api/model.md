# Model Interpretation using SHAP

The aim of this module is to provide tools for model interpretation using the [SHAP](https://shap.readthedocs.io/en/latest/) library.
The classes below provide convenient wrappers that implement multiple visualization techniques for model interpretation.

## Overview

The model interpretation module provides comprehensive tools for understanding and visualizing machine learning models using SHAP (SHapley Additive exPlanations). This module is particularly useful for:

- Analyzing feature importance and their impact on model predictions
- Understanding model behavior through various visualization techniques
- Interpreting both tree-based and linear models
- Examining individual predictions and feature contributions
- Detecting feature interactions and dependencies

### Key Components

#### ShapModelInterpreter
- A comprehensive wrapper for SHAP analysis
- Provides multiple visualization options:
  - Feature importance bar plots
  - SHAP summary beeswarm plots
  - Dependence plots for feature interaction analysis
  - Waterfall plots for individual prediction explanation
- Supports both classification and regression models
- Handles multi-class classification scenarios
- Includes model performance metrics
- Compatible with sklearn Pipeline objects for seamless preprocessing integration

#### DependencePlotter
- Specialized tool for creating SHAP dependence plots
- Creates dual-panel visualizations showing:
  - SHAP values vs. feature values
  - Feature value distribution and target rate trends
- Helps visualize how individual features affect model predictions
- Supports both classification and regression models
- Provides extensive customization options:
  - Quantile-based data filtering for outlier handling
  - Binning controls for target rate visualization
  - Customizable transparency and plot styling

### Features
- Automatic handling of different model types
- Support for binary classification, multi-class classification, and regression
- Flexible data preprocessing options:
  - Custom feature naming
  - Class labeling options
  - Pre-calculated SHAP value support
- Customizable visualization options:
  - Plot styling and dimensions
  - Feature subset selection
  - Multiple plot types for different analysis needs
- Performance monitoring with train/test metrics
- SHAP variance penalization for more stable importance calculations

### Use Cases
- Model debugging and validation
- Feature importance analysis
- Feature interaction discovery
- Individual prediction explanation
- Model comparison and selection
- Stakeholder communication
- Regulatory compliance and model documentation
- Model optimization and feature selection

The module is designed to make model interpretation accessible and comprehensive, providing both high-level insights and detailed analysis capabilities through visually rich and informative plots.

## Implementation

::: probatus.model.shap_interpreter
::: probatus.model.shap_dependence_plotter