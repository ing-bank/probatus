# TODO: Make this a base class which contains:
# - SHAP explainer
# - SHAP explanation
# - SHAP values
# - SHAP expected value

# From here all the other classes interact with outside.
# This'll be a SHAP object that is thus a wrapper around all SHAP related values.

import pandas as pd
import shap


class ShapObject:
    # TODO: For example add aggregated values but also original. Something like SHAP_values, SHAP_values_aggregated, SHAP_values_original
    def __init__(
        self, explainer: shap.Explainer, explanation: shap.Explanation, values: pd.DataFrame, expected_value: float
    ):
        self.explainer = explainer
        self.explanation = explanation
        self.values = values
        self.expected_value = expected_value
