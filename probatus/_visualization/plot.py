from typing import Literal, Optional, Any, Dict, Union, List

from matplotlib import pyplot as plt
import shap

from probatus._visualization.plot_enum import PlotTypeEnum
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.shap_new.manager import SHAPManager
from probatus.wrapper.shap_new.instance import SHAPInstance


class PlotManager:
    def __init__(
        self,
        shap_manager: SHAPManager,
        data_manager: BaseDataManager,
        verbose: Literal[0, 1, 2] = 0,
    ):
        self.shap_manager = shap_manager
        self.data_manager = data_manager
        self.verbose = verbose

    def plot(
        self,
        shap_instance: SHAPInstance,
        plot_type: PlotTypeEnum,
        split_selection: Literal["full", "train", "test"] = "test",
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        show: bool = False,
        **kwargs,
    ) -> plt.Figure | List[plt.Figure]:
        aggregated_values = self.shap_manager.get_aggregated_values(
            shap_instance,
            self.data_manager,
            split_selection,
            class_selection,
            weights,
            multi_class_aggregation,
            shap_variance_penalty_factor,
            self.verbose,
        )

        # Prepare data for plotting
        explanation = shap.Explanation(
            values=aggregated_values,
            feature_names=self.data_manager.column_names,
        )

        # Prepare styling

        # Create plot

        # Show plot (or not)

        # Return plot
        pass
