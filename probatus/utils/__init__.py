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

from .exceptions import NotFittedError, DimensionalityError, UnsupportedModelError
from .scoring import Scorer, get_scorers
from .arrayfuncs import assure_numpy_array, assure_pandas_df, check_1d, warn_if_missing, check_numeric_dtypes, \
    assure_column_names_consistency
from .shap_helpers import shap_calc, shap_to_df, calculate_shap_importance
from .warnings import ApproximationWarning
from ._utils import class_name_from_object, assure_list_of_strings, assure_list_values_allowed
from .tree import TreePathFinder
from .plots import plot_distributions_of_feature

__all__ = ['NotFittedError', 'DimensionalityError', 'UnsupportedModelError', 'Scorer', 'assure_numpy_array', 'assure_pandas_df', 'check_1d',
           'ApproximationWarning', 'class_name_from_object', 'get_scorers', 'assure_list_of_strings',
           'assure_list_values_allowed', 'warn_if_missing','TreePathFinder', 'check_numeric_dtypes',
           'plot_distributions_of_feature', 'assure_column_names_consistency', 'shap_calc', 'shap_to_df', 'calculate_shap_importance']
