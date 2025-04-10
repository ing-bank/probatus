from enum import Enum

import shap


class PlotTypeEnum(str, Enum):
    # bar = "bar"
    # beeswarm = "beeswarm"
    # waterfall = "waterfall"
    # distribution = "distribution"
    # interaction = "interaction"
    # scatter = "scatter"
    # importance = "importance"
    # summary = "summary"
    # dependence = "dependence"
    # sample = "sample"
    # decision = "decision"
    permutation = "permutation"

    def __call__(self, *args, **kwargs):
        # TODO: Also add support for custom plots & multi-class plots
        return getattr(shap.plots, self.value)(*args, **kwargs)
