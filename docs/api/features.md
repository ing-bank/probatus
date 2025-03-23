# Features Elimination

This module focuses on feature elimination and it contains the following:

- [ShapRFECV][probatus.feature_elimination.feature_elimination.ShapRFECV]: Perform Backwards Recursive Feature Elimination, using SHAP feature importance. It supports binary classification, multi-class classification, regression models and hyperparameter optimization at every feature elimination step. For XGBoost, LightGBM and CatBoost it supports early stopping of the model fitting process. It can be an alternative regularization technique to hyperparameter optimization of the number of base trees in gradient boosted tree models. Particularly useful when dealing with large datasets.

## Overview

The feature elimination module provides sophisticated tools for selecting the most important features in machine learning models using SHAP-based importance metrics. This module is particularly valuable for:

- Reducing model complexity while maintaining performance
- Identifying the most impactful features
- Optimizing model training time
- Improving model interpretability

### Key Components

#### ShapRFECV (Recursive Feature Elimination with Cross-Validation)
- Implements backward feature elimination using SHAP importance
- Key features:
  - Supports both classification and regression tasks
  - Integrates with hyperparameter optimization via SearchCV, RandomizedSearchCV, or BayesSearchCV, etc...
  - Provides cross-validation at each elimination step
  - Generates detailed feature selection reports
  - Supports custom feature selection strategies
  - Includes advanced methods to select optimal feature subsets:
    - Best performance-based selection
    - Coherent performance with minimal variance
    - Parsimonious selection (fewest features within a performance threshold)
  - Supports sklearn Pipeline objects for seamless preprocessing integration

#### Early Stopping Support
- Directly integrated into ShapRFECV (no separate class needed)
- Optimized for popular gradient boosting frameworks:
  - LightGBM
  - XGBoost
  - CatBoost
- Benefits:
  - Faster training process
  - Prevents overfitting
  - Reduces computational resources
  - Maintains model performance

### Features
- Flexible feature elimination strategies:
  - Fixed number of features per step
  - Percentage-based elimination (adaptive step sizes)
  - Custom elimination criteria
- Comprehensive model support:
  - Binary classification
  - Multi-class classification
  - Regression models
- Advanced optimization capabilities:
  - Hyperparameter tuning integration
  - Cross-validation at each step
  - Early stopping for gradient boosted models
  - Support for sample weighting
  - Optional SHAP variance penalization for more stable feature selection
- Detailed reporting and visualization:
  - Feature importance tracking
  - Performance metrics across steps
  - Selection process visualization
  - Error bars showing model stability

### Use Cases
- Large-scale feature selection
- Model optimization
- Feature importance analysis
- Computational resource optimization
- Model interpretability enhancement
- Production model preparation

The module is designed to provide a robust and efficient approach to feature selection using RFECV in combination with SHAP, with a focus on practical application in real-world scenarios.

## Implementation

::: probatus.features.shap_recursive_feature_elimination
::: probatus.features.shap_recursive_feature_elimination_helper
::: probatus.features.shap_early_stopping_recursive_feature_elimination
::: probatus.features.shap_early_stopping_recursive_feature_elimination_helper
