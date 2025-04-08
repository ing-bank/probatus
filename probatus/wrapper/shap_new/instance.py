import pandas as pd
from probatus.wrapper.shap_new.parameters import extract_shap_parameters
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel
from shap.utils import sample
from shap.explainers import TreeExplainer
import numpy as np
import shap


from typing import Literal, Optional


class SHAPInstance:
    def __init__(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        random_state: Optional[int] = None,
        **shap_kwargs,
    ):
        self.explainer: shap.Explainer = self._create_explainer(
            model=model,
            data_manager=data_manager,
            split_selection=split_selection,
            random_state=random_state,
            **shap_kwargs,
        )
        self.expected_value: np.ndarray = self.explainer.expected_value
        self.explanation: shap.Explanation = SHAPInstance._calculate_explanation(
            data_manager=data_manager, split_selection=split_selection, **shap_kwargs
        )
        self.values: np.ndarray = self.explanation.values

    @staticmethod
    def _create_explainer(
        model: BaseModel,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        random_state: Optional[int] = None,
        sample_size: int = 100,
        **shap_kwargs,
    ) -> shap.Explainer:
        # Split arguments for multi-classification
        _, shap_explainer_kwargs, _ = extract_shap_parameters(shap_kwargs)

        # Check if the dataset has categorical features
        has_categorical = False
        if isinstance(data_manager.get_X(split_selection), pd.DataFrame):
            has_categorical = data_manager.get_X(split_selection).select_dtypes("category").shape[1] > 0

        # For non-tree models, we reserve a background sample (masker) to provide a realistic baseline for
        # perturbing features, since tree-based models inherently manage feature variations. Perturbing features
        # quantify each feature's contribution relative to this baseline, which is key to SHAP calculations.
        masker = None
        if not has_categorical and shap_kwargs.get("feature_perturbation") != "tree_path_dependent":
            # If the dataset is smaller than the requested sample size, use a percentage of the dataset
            if data_manager.get_X(split_selection).shape[0] < sample_size:
                sample_size = int(np.ceil(data_manager.get_X(split_selection).shape[0] * 0.2))  # Use 20% of the dataset

            # Create background data by sampling from the input dataset
            masker = sample(data_manager.get_X(split_selection), sample_size, random_state=random_state)

        # such as the approximate parameter
        return shap.Explainer(
            model=model.estimator,
            masker=masker,
            output_names=data_manager.class_names,
            feature_names=data_manager.column_names,
            seed=random_state,
            **shap_explainer_kwargs,
        )

    @staticmethod
    def _calculate_explanation(
        explainer: shap.Explainer,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        **shap_kwargs,
    ) -> shap.Explanation:
        # Filter out approximate parameter
        _, _, shap_tree_explanation_kwargs = extract_shap_parameters(shap_kwargs)

        if isinstance(explainer, TreeExplainer):
            # Tree-based models can use approximation for faster calculation
            return explainer(data_manager.get_X(split_selection), **shap_tree_explanation_kwargs)
        else:
            return explainer(data_manager.get_X(split_selection))
