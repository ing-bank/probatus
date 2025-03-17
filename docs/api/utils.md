# Utility Functions

This module contains various smaller functionalities that can be used across the `probatus` package.

## Overview

The package contains several utility modules that provide helpful functionalities across different parts of the codebase:

### probatus.utils.scoring
- Provides functionality for model evaluation and scoring
- Key components:
  - `Scorer` class: A wrapper for model evaluation metrics
  - `get_single_scorer()`: Standardizes scoring metrics for consistent model evaluation
- Supports both predefined scikit-learn metrics and custom metrics

### probatus.utils.array
- Handles data preprocessing and array manipulation
- Key functions:
  - `assure_pandas_df()`: Ensures data is in pandas DataFrame format
  - `assure_pandas_series()`: Ensures data is in pandas Series format
  - `preprocess_data()`: Preprocesses input data for model training
  - `preprocess_labels()`: Preprocesses target labels

### probatus.utils.common
- Contains general-purpose utility functions
- Key functions:
  - `assure_list_of_strings()`: Ensures input is a list of strings
  - `is_regression_model()`: Checks if a model is a regression model

### probatus.utils.shap
- Provides utilities for SHAP (SHapley Additive exPlanations) analysis
- Key functions:
  - `shap_calc()`: Calculates SHAP values for a model
  - `shap_to_df()`: Converts SHAP values to a pandas DataFrame
  - `calculate_shap_importance()`: Computes feature importance based on SHAP values
- Includes validation and formatting functions for SHAP analysis
- Supports both tree-based and non-tree models
- Handles multi-class and binary classification cases

These utility functions are used throughout the package to:
- Ensure consistent data formats
- Handle model evaluation
- Process and validate inputs
- Calculate feature importance
- Support SHAP analysis for model interpretation

The utilities are designed to be modular and reusable, making it easier to maintain consistent behavior across different parts of the package while reducing code duplication.

## Implementation

::: probatus.utils.scoring
::: probatus.utils.array
::: probatus.utils.common
::: probatus.utils.shap
