"""
Test that legacy imports still work correctly.
"""

from probatus.data_comparison._base import BaseResemblanceModel
from probatus.data_comparison.permutation.importance import PermutationImportanceResemblance


# TODO: Add legacy imports for renamed methods


def test_legacy_sample_similarity_imports():
    """Test that legacy sample_similarity imports work."""
    from probatus.sample_similarity import (
        BaseResemblanceModel,
        PermutationImportanceResemblance,
        SHAPImportanceResemblance,
    )

    assert BaseResemblanceModel.__module__ == "probatus.sample_similarity"
    assert PermutationImportanceResemblance.__module__ == "probatus.sample_similarity"
    assert SHAPImportanceResemblance.__module__ == "probatus.sample_similarity"


def test_legacy_interpret_imports():
    """Test that legacy interpret imports work."""
    from probatus.model_interpretation import ShapModelInterpreter, ShapDependencePlotter

    assert ShapModelInterpreter.__module__ == "probatus.interpret"
    assert ShapDependencePlotter.__module__ == "probatus.interpret"


def test_legacy_feature_elimination_imports():
    """Test that legacy feature_elimination imports work."""
    from probatus.feature_elimination import ShapRFECV, EarlyStoppingShapRFECV

    assert ShapRFECV.__module__ == "probatus.feature_elimination"
    assert EarlyStoppingShapRFECV.__module__ == "probatus.feature_elimination"


def test_new_imports():
    """Test that new imports work correctly."""
    from probatus._core import BaseFitComputeClass, BaseFitComputePlotClass
    from probatus.data_comparison.shap.importance import (
        ShapImportanceResemblance,
    )
    from probatus.model_interpretation import ShapModelInterpreter, ShapDependencePlotter
    from probatus.features import ShapRFECV

    # Just verify that imports work, no need to check module names as these are the actual locations
    assert BaseFitComputeClass  # type: ignore[truthy-function]
    assert BaseFitComputePlotClass  # type: ignore[truthy-function]
    assert BaseResemblanceModel  # type: ignore[truthy-function]
    assert PermutationImportanceResemblance  # type: ignore[truthy-function]
    assert ShapImportanceResemblance  # type: ignore[truthy-function]
    assert ShapModelInterpreter  # type: ignore[truthy-function]
    assert ShapDependencePlotter  # type: ignore[truthy-function]
    assert ShapRFECV  # type: ignore[truthy-function]
