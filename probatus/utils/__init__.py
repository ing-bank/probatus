# Copyright (c) 2020 ING Bank N.V.
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

from ._utils import assure_list_of_strings, assure_list_values_allowed, class_name_from_object
from .arrayfuncs import (
    assure_numpy_array,
    assure_pandas_df,
    assure_pandas_series,
    check_1d,
    check_numeric_dtypes,
    preprocess_data,
    preprocess_labels,
)
from .exceptions import DimensionalityError, NotFittedError, NotInstalledError, UnsupportedModelError
from .interface import BaseFitComputeClass, BaseFitComputePlotClass
from .plots import plot_distributions_of_feature
from .scoring import Scorer, get_scorers, get_single_scorer
from .shap_helpers import calculate_shap_importance, shap_calc, shap_to_df
from .warnings import ApproximationWarning

__all__ = [
    "NotFittedError",
    "DimensionalityError",
    "UnsupportedModelError",
    "NotInstalledError",
    "Scorer",
    "assure_numpy_array",
    "assure_pandas_df",
    "check_1d",
    "ApproximationWarning",
    "class_name_from_object",
    "get_scorers",
    "assure_list_of_strings",
    "assure_list_values_allowed",
    "check_numeric_dtypes",
    "plot_distributions_of_feature",
    "shap_calc",
    "shap_to_df",
    "calculate_shap_importance",
    "assure_pandas_series",
    "preprocess_data",
    "preprocess_labels",
    "BaseFitComputeClass",
    "BaseFitComputePlotClass",
    "get_single_scorer",
]
