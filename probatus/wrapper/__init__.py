from ._shap import (
    calculate_shap_explanation,
    shap_explanation_to_shap_values,
    calculate_shap_importance_dataframe,
    extract_multi_class_shap_parameters,
    process_shap_values,
    calculate_base_shap_statistics,
    create_importance_dataframe,
    aggregate_multiclass_shap_values_values,
    calculate_shap_and_expected_values,
)

from .exceptions import NotFittedError
from .base import BaseFitComputeClass, BaseFitComputePlotClass
from .scoring import Scorer, get_single_scorer

__all__ = [
    "calculate_shap_explanation",
    "shap_explanation_to_shap_values",
    "calculate_shap_importance_dataframe",
    "extract_multi_class_shap_parameters",
    "process_shap_values",
    "calculate_base_shap_statistics",
    "create_importance_dataframe",
    "aggregate_multiclass_shap_values_values",
    "calculate_shap_and_expected_values",
    "NotFittedError",
    "BaseFitComputeClass",
    "BaseFitComputePlotClass",
    "get_single_scorer",
    "Scorer",
]
