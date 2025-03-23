# Utility Functions

This module contains various reusable functionalities that support the core components of the `probatus` package.

## Overview

The utils module provides several specialized submodules that offer consistent data handling, model evaluation, and SHAP analysis across different parts of the library:

### probatus.utils.scoring
- Provides standardized model evaluation metrics
- Key components:
  - `Scorer` class: A flexible wrapper for both built-in and custom evaluation metrics
  - `get_single_scorer()`: Ensures consistent metric handling throughout the package
- Supports both classification and regression metrics
- Enables custom metric definition through scikit-learn's make_scorer

### probatus.utils.array
- Handles data preprocessing and format conversion
- Key functions:
  - `assure_pandas_df()`: Converts various data types to pandas DataFrame
  - `assure_pandas_series()`: Converts various data types to pandas Series with index handling
  - `preprocess_data()`: Prepares feature data with warnings for missing values and categorical features
  - `preprocess_labels()`: Ensures target data has proper format and indexing

### probatus.utils.common
- Provides general-purpose utilities for common operations
- Key functions:
  - `assure_list_of_strings()`: Validates and converts string inputs to lists
  - `is_regression_model()`: Detects if a model is for regression vs. classification
  - `handle_class_names()`: Manages class name formatting for visualization
  - `get_pipeline_preprocessor_and_estimator()`: Extracts components from scikit-learn pipelines
  - `preprocess_using_pipeline()`: Applies pipeline preprocessing to new data

### probatus.utils.shap
- Delivers comprehensive SHAP (SHapley Additive exPlanations) analysis tools
- Key functions:
  - `calculate_shap_explanation()`: Computes SHAP values with appropriate explainer type selection
  - `shap_explanation_to_shap_df()`: Formats SHAP values into pandas DataFrame
  - `calculate_shap_importance()`: Computes feature importance metrics from SHAP values
  - `extract_shap_multiclass_params()`: Separates multiclass parameters from general SHAP parameters
  - `prep_shap_related_variables()`: Prepares SHAP values and extracts the expected value
- Advanced features:
  - Pipeline compatibility for preprocessing integration
  - Multi-class support with various aggregation strategies:
    - max_abs: Highlights most influential features across any class
    - variance: Identifies features with varying impact between classes
    - mean_abs: Provides balanced importance across all classes
  - Custom class weighting for imbalanced datasets
  - Variance penalization for more stable feature importance
  - Background data sampling for consistent baseline calculations

These utilities serve several key purposes in the probatus package:

1. **Data Consistency**: Ensuring uniform data formats across different components
2. **Evaluation Standardization**: Providing consistent model performance metrics
3. **Model Compatibility**: Supporting various model types and scikit-learn pipelines
4. **SHAP Analysis**: Offering flexible and robust model interpretation capabilities
5. **Error Handling**: Providing clear error messages and warnings for common issues

The modular design enables these functions to be used independently or combined for complex analytical workflows, while maintaining consistent behavior across the package.

## Implementation

::: probatus.utils.scoring
::: probatus.utils.array
::: probatus.utils.common
::: probatus.utils.shap
