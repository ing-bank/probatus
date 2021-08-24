# Features Elimination

This module focuses on feature elimination and it contains two classes:

- [ShapRFECV][probatus.feature_elimination.feature_elimination.ShapRFECV]: Perform Backwards Recursive Feature Elimination, using SHAP feature importance. It supports binary classification models and hyperparameter optimization at every feature elimination step.
- [EarlyStoppingShapRFECV][probatus.feature_elimination.feature_elimination.EarlyStoppingShapRFECV]: adds support to early stopping of the model fitting process. It can be an alternative regularization technique to hyperparameter optimization of the number of base trees in gradient boosted tree models. Particularly useful when dealing with large datasets.

::: probatus.feature_elimination.feature_elimination
