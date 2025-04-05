import warnings
import shap
import numpy as np
from typing import Any, Dict, Literal, Optional, Union, List

from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel


class SHAPInstance:
    def __init__(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        random_state: Optional[int] = None,
        **shap_kwargs,
    ):
        self.explainer: shap.Explainer = self._create_explainer(
            model=model, data_manager=data_manager, random_state=random_state, **shap_kwargs
        )
        self.expected_value: np.ndarray = self.explainer.expected_value
        self.explanation: shap.Explanation = SHAPInstance._calculate_explanation(
            data_manager=data_manager, **shap_kwargs
        )
        self.values: np.ndarray = self.explanation.values

    @staticmethod
    def _create_explainer(
        model: BaseModel, data_manager: BaseDataManager, random_state: Optional[int] = None, **shap_kwargs
    ) -> shap.Explainer:
        # TODO: Fix explainer creation; filter out kwargs that are not valid for the explainer
        # such as the approximate parameter
        return shap.Explainer(
            model=model.model,
            masker=data_manager.X,
            output_names=data_manager.class_names,
            feature_names=data_manager.column_names,
            seed=random_state,
            **shap_kwargs,
        )

    @staticmethod
    def _calculate_explanation(
        explainer: shap.Explainer,
        data_manager: BaseDataManager,
        **shap_kwargs,
    ) -> shap.Explanation:
        # Create explanation object without recalculating shap_values
        return explainer(data_manager.X, **shap_kwargs)


class SHAPManager:
    def __init__(
        self,
        random_state: Optional[int] = None,
    ):
        self.random_state: int = random_state
        self.shap_instances: dict[tuple, SHAPInstance] = {}
        self.aggregated_values_cache: dict[tuple, np.ndarray] = {}

    def get_shap_instance(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        random_state: Optional[int] = None,
        verbose: Literal[0, 1, 2] = 0,
        cache: bool = True,
    ) -> SHAPInstance:
        # Create SHAP key, based on only the feature names
        shap_instance_key = SHAPManager._create_shap_instance_key(data_manager)

        if cache:
            # Add SHAP instance to the cache & use cache if it exists
            if shap_instance_key not in self.shap_instances:
                if verbose > 0:
                    warnings.warn(
                        f"SHAP instance not found in cache. Calculating SHAP instance for key: {shap_instance_key}"
                    )

                # Create SHAP instance
                self.shap_instances[shap_instance_key] = SHAPInstance(model, data_manager, random_state)

            return self.shap_instances[shap_instance_key]
        else:
            return SHAPInstance(model, data_manager, random_state)

    def get_aggregated_values(
        self,
        explanation: shap.Explanation,
        data_manager: BaseDataManager,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        verbose: Literal[0, 1, 2] = 0,
        cache: bool = True,
    ) -> np.ndarray:
        # Create a key for the aggregated values
        key = SHAPManager._create_aggregation_values_key(
            data_manager=data_manager,
            class_selection=class_selection,
            weights=weights,
            multi_class_aggregation=multi_class_aggregation,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        if cache:
            # If the key is not in the aggregated values dictionary, calculate the aggregated values
            if key not in self.aggregated_values_cache:
                if verbose > 0:
                    warnings.warn(f"Aggregation not found in cache. Calculating aggregated values for key: {key}")

                # Calculate the aggregated values
                self.aggregated_values_cache[key] = SHAPManager._calculate_aggregated_values(
                    explanation=explanation,
                    class_selection=class_selection,
                    weights=weights,
                    multi_class_aggregation=multi_class_aggregation,
                    shap_variance_penalty_factor=shap_variance_penalty_factor,
                )

            return self.aggregated_values_cache[key]
        else:
            return SHAPManager._calculate_aggregated_values(
                explanation=explanation,
                class_selection=class_selection,
                weights=weights,
                multi_class_aggregation=multi_class_aggregation,
                shap_variance_penalty_factor=shap_variance_penalty_factor,
            )

    def get_class_values(
        self,
        explanation: shap.Explanation,
        data_manager: BaseDataManager,
        class_selection: Optional[Any] = None,
        verbose: Literal[0, 1, 2] = 0,
        cache: bool = True,
    ) -> dict[str, np.ndarray]:
        # Create a key for the aggregated values
        key = SHAPManager._create_aggregation_values_key(
            data_manager=data_manager,
            class_selection=class_selection,
            weights=None,
            multi_class_aggregation=None,
            shap_variance_penalty_factor=0,
        )

        if cache:
            # If the key is not in the aggregated values dictionary, calculate the aggregated values
            if key not in self.aggregated_values_cache:
                if verbose > 0:
                    warnings.warn(f"Aggregation not found in cache. Calculating aggregated values for key: {key}")

                # Calculate the aggregated values
                self.aggregated_values_cache[key] = SHAPManager._calculate_aggregated_values(
                    explanation=explanation,
                    class_selection=class_selection,
                    weights=None,
                    multi_class_aggregation=None,
                    shap_variance_penalty_factor=0,
                )

            return self.aggregated_values_cache[key]
        else:
            return SHAPManager._calculate_aggregated_values(
                explanation=explanation,
                class_selection=class_selection,
                weights=None,
                multi_class_aggregation=None,
                shap_variance_penalty_factor=0,
            )

    @staticmethod
    def _calculate_aggregated_values(
        explanation: shap.Explanation,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Optional[Union[int, float]] = 0,
    ) -> np.ndarray:
        # TODO: fix the aggregation function
        # At the very least requires aggregation across ...
        return np.sum(explanation.values, axis=0)

    @staticmethod
    def _create_shap_instance_key(
        data_manager: BaseDataManager,
    ) -> str:
        return SHAPManager._convert_feature_names_list_to_str(data_manager)

    @staticmethod
    def _create_aggregation_values_key(
        data_manager: BaseDataManager,
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = None,
    ) -> tuple[
        List[str],  # Class names
        Union[str, int],  # Class selection
        str,  # Weights (dict formatted as a ordered string)
        Union[str, Literal["mean", "mean_abs", "max_abs"]],  # Aggregation method
        Union[str, float, int],  # Penalty factor
    ]:
        # Convert complex objects to strings
        class_names_str = SHAPManager._convert_feature_names_list_to_str(data_manager)
        weights_str = SHAPManager._convert_weights_dict_to_str(weights)

        # Create a key for the aggregated values
        return (
            class_names_str,
            class_selection if class_selection else "_",
            weights_str if weights_str else "_",
            multi_class_aggregation if multi_class_aggregation else "_",
            shap_variance_penalty_factor if shap_variance_penalty_factor else 0,
        )

    @staticmethod
    def _convert_weights_dict_to_str(weights: dict[Any, float]) -> str:
        # First order the dictionary by key
        sorted_dictionary = dict(sorted(weights.items()))

        # Then convert to a string
        return "".join(f"{k}:{v}" for k, v in sorted_dictionary.items())

    @staticmethod
    def _convert_feature_names_list_to_str(data_manager: BaseDataManager) -> str:
        return "_-_".join(data_manager.column_names)
