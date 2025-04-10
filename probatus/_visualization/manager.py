from typing import Literal, Optional, Any, Dict, Union, List

from matplotlib import pyplot as plt

from probatus._visualization._base_plot import BasePlot
from probatus._visualization.enum import PlotTypeEnum
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel
from probatus.wrapper.shap_new.manager import SHAPManager
from probatus.wrapper.shap_new.instance import SHAPInstance


class PlotManager:
    def __init__(
        self,
        model: BaseModel,
        shap_manager: SHAPManager,
        data_manager: BaseDataManager,
        verbose: Literal[0, 1, 2] = 0,
        show: bool = False,
        **common_plot_kwargs,
    ):
        self.model = model
        self.shap_manager = shap_manager
        self.data_manager = data_manager
        self.verbose = verbose
        self.show = show
        # TODO: Define default kwargs properties that should always appear on any plot
        # Currently not used by create_plot
        self.common_plot_kwargs = common_plot_kwargs

    def create_plot(
        self,
        shap_instance: SHAPInstance,
        plot_type: PlotTypeEnum,
        split_selection: Literal["full", "train", "test"] = "test",
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        **plot_kwargs,
    ) -> plt.Figure | List[plt.Figure]:
        # TODO: Filter out different plot_kwargs per step
        plot_instance = BasePlot(
            model=self.model,
            data_manager=self.data_manager,
            show=self.show,
            **plot_kwargs,
        )

        return plot_instance.plot(
            shap_instance,
            plot_type,
            split_selection,
            class_selection,
            weights,
            multi_class_aggregation,
            shap_variance_penalty_factor,
            show=self.show,
            **plot_kwargs,
        )
