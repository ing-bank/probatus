# ruff: noqa
# mypy: ignore-errors
# Copyright (c) ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

name = "probatus"

# TODO: Deprecate the old exports in a future release.
# ===== BACKWARDS COMPATIBILITY IMPORTS =====
# This section ensures that legacy imports continue to work after the package
# structure has been reorganized. It creates dummy modules for old import paths
# and maps the new classes to them.

import sys

from probatus.data_comparison._base import BaseResemblanceModel
from probatus.data_comparison.permutation.importance import PermutationImportanceResemblance

# Create legacy module paths as dummy modules
sys.modules["probatus.sample_similarity"] = type("LegacyModule", (), {})
sys.modules["probatus.interpret"] = type("LegacyModule", (), {})
sys.modules["probatus.feature_elimination"] = type("LegacyModule", (), {})

# Import directly from implementation files to avoid circular imports
from probatus.wrapper.base import BaseFitComputeClass, BaseFitComputePlotClass
from probatus.data_comparison.shap.importance import (
    ShapImportanceResemblance,
)
from probatus.model_interpretation.interpreter import ShapModelInterpreter
from probatus.model_interpretation.dependence_plotter import ShapDependencePlotter
from probatus.features.feature_elimination.rfe import ShapRFECV

# Map classes to their legacy module paths
sys.modules["probatus.sample_similarity"].BaseResemblanceModel = BaseResemblanceModel
sys.modules["probatus.sample_similarity"].PermutationImportanceResemblance = PermutationImportanceResemblance
sys.modules["probatus.sample_similarity"].SHAPImportanceResemblance = ShapImportanceResemblance
sys.modules["probatus.interpret"].ShapModelInterpreter = ShapModelInterpreter
sys.modules["probatus.interpret"].DependencePlotter = ShapDependencePlotter
sys.modules["probatus.feature_elimination"].ShapRFECV = ShapRFECV
sys.modules["probatus.feature_elimination"].EarlyStoppingShapRFECV = ShapRFECV

# Set the module attributes for proper imports
sys.modules["probatus.sample_similarity"].BaseResemblanceModel.__module__ = "probatus.sample_similarity"
sys.modules["probatus.sample_similarity"].PermutationImportanceResemblance.__module__ = "probatus.sample_similarity"
sys.modules["probatus.sample_similarity"].SHAPImportanceResemblance.__module__ = "probatus.sample_similarity"

sys.modules["probatus.interpret"].ShapModelInterpreter.__module__ = "probatus.interpret"
sys.modules["probatus.interpret"].DependencePlotter.__module__ = "probatus.interpret"

sys.modules["probatus.feature_elimination"].ShapRFECV.__module__ = "probatus.feature_elimination"
sys.modules["probatus.feature_elimination"].EarlyStoppingShapRFECV.__module__ = "probatus.feature_elimination"

# Define public API
__all__ = [
    "BaseFitComputeClass",
    "BaseFitComputePlotClass",
    "BaseResemblanceModel",
    "PermutationImportanceResemblance",
    "ShapImportanceResemblance",
    "ShapModelInterpreter",
    "ShapDependencePlotter",
    "ShapRFECV",
    "EarlyStoppingShapRFECV",
]
