import warnings
import shap
import numpy as np
from typing import Any, Dict, Literal, Optional, Union

from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel


class SHAPManager:
    def __init__(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        random_state: Optional[int] = None,
        **shap_kwargs,
    ):
        self.random_state: int = random_state
        self.explainer: shap.Explainer = self._create_explainer(
            model=model, data_manager=data_manager, random_state=random_state, **shap_kwargs
        )
        self.expected_value: np.ndarray = self.explainer.expected_value

    def calculate_parameters(
        self,
        data_manager: BaseDataManager,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Optional[Union[int, float]] = None,
        **shap_kwargs,
    ) -> None:
        self.explanation: shap.Explanation = SHAPManager._calculate_explanation(
            data_manager=data_manager, **shap_kwargs
        )

        # Aggregation is expensive, so we cache the results in a dictionary.
        # Initially starts with 0 or 1 aggregation.
        self.aggregated_values: dict[tuple, np.ndarray] = SHAPManager._init_aggregated_values(
            explanation=self.explanation,
            class_selection=class_selection,
            weights=weights,
            multi_class_aggregation=multi_class_aggregation,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

    def _calculate_explanation(
        self,
        data_manager: BaseDataManager,
        **shap_kwargs,
    ) -> shap.Explanation:
        # Create explanation object without recalculating shap_values
        return self.explainer(data_manager.X, **shap_kwargs)

    @staticmethod
    def _create_explainer(model: BaseModel, data_manager: BaseDataManager, **shap_kwargs) -> shap.Explainer:
        # TODO: Fix explainer creation; filter out kwargs that are not valid for the explainer
        # such as the approximate parameter
        return shap.Explainer(
            model=model.model,
            masker=data_manager.X,
            output_names=data_manager.class_names,
            feature_names=data_manager.column_names,
            **shap_kwargs,
        )

    @staticmethod
    def _calculate_aggregated_values(
        values: np.ndarray,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        # TODO: fix the aggregation function
        # At the very least requires aggregation across ...
        return np.sum(values, axis=0)

    def get_aggregated_values(
        self,
        explanation: shap.Explanation,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        verbose: Literal[0, 1, 2] = 0,
    ) -> np.ndarray:
        # Create a key for the aggregated values
        key = SHAPManager._create_key(
            class_selection=class_selection,
            weights=weights,
            multi_class_aggregation=multi_class_aggregation,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        # If the key is not in the aggregated values dictionary, calculate the aggregated values
        if key not in self.aggregated_values:
            if verbose > 0:
                warnings.warn(f"Aggregation not found in cache. Calculating aggregated values for key: {key}")
            self.aggregated_values[key] = SHAPManager._calculate_aggregated_values(
                values=explanation.values,
                class_selection=class_selection,
                weights=weights,
                multi_class_aggregation=multi_class_aggregation,
                shap_variance_penalty_factor=shap_variance_penalty_factor,
            )
            return self.aggregated_values[key]

    def get_class_values(
        self, explanation: shap.Explanation, class_selection: Optional[Any] = None, verbose: Literal[0, 1, 2] = 0
    ) -> dict[str, np.ndarray]:
        # Create a key for the aggregated values
        key = SHAPManager._create_key(
            class_selection=class_selection,
            weights=None,
            multi_class_aggregation=None,
            shap_variance_penalty_factor=None,
        )

        # If the key is not in the aggregated values dictionary, calculate the aggregated values
        if key not in self.aggregated_values:
            if verbose > 0:
                warnings.warn(f"Aggregation not found in cache. Calculating aggregated values for key: {key}")
            self.aggregated_values[key] = SHAPManager._calculate_aggregated_values(
                values=explanation.values,
                class_selection=class_selection,
                weights=None,
                multi_class_aggregation=None,
                shap_variance_penalty_factor=None,
            )
            return self.aggregated_values[key]

    @staticmethod
    def _init_aggregated_values(
        explanation: shap.Explanation,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
    ) -> dict[tuple, np.ndarray]:
        # Initialize the aggregated values dictionary
        aggregated_values: dict[
            tuple[
                Union[str, int],  # Class selection
                str,  # Weights (dict formatted as a ordered string)
                Union[str, Literal["mean", "mean_abs", "max_abs"]],  # Aggregation method
                Union[str, float, int],  # Penalty factor
            ],
            np.ndarray,
        ] = {}

        # Create a key for the aggregated values
        key = SHAPManager._create_key(
            class_selection=class_selection,
            weights=weights,
            multi_class_aggregation=multi_class_aggregation,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        # Calculate the aggregated values
        aggregated_values[key] = SHAPManager._calculate_aggregated_values(
            values=explanation.values,
            class_selection=class_selection,
            weights=weights,
            multi_class_aggregation=multi_class_aggregation,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        # Return the dictionary
        return aggregated_values

    @staticmethod
    def _create_key(
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = None,
    ) -> tuple[
        Union[str, int],  # Class selection
        str,  # Weights (dict formatted as a ordered string)
        Union[str, Literal["mean", "mean_abs", "max_abs"]],  # Aggregation method
        Union[str, float, int],  # Penalty factor
    ]:
        # Convert the weights dictionary to a string
        weights_str = SHAPManager._convert_dict_to_str(weights)

        # Create a key for the aggregated values
        return (
            class_selection if class_selection else "_",
            weights_str if weights_str else "_",
            multi_class_aggregation if multi_class_aggregation else "_",
            shap_variance_penalty_factor if shap_variance_penalty_factor else "_",
        )

    @staticmethod
    def _convert_dict_to_str(dictionary: dict[Any, float]) -> str:
        # First order the dictionary by key
        sorted_dictionary = dict(sorted(dictionary.items()))

        # Then convert to a string
        return "".join(f"{k}:{v}" for k, v in sorted_dictionary.items())
