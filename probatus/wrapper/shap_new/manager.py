import shap
from probatus.wrapper.data import BaseDataManager
from probatus.wrapper.estimator import BaseModel
from probatus.wrapper.shap_new.aggregation import calculate_aggregated_values
from probatus.wrapper.shap_new.cache import create_aggregation_values_key, create_shap_instance_key
from probatus.wrapper.shap_new.instance import SHAPInstance


import numpy as np


import warnings
from typing import Any, Dict, Literal, Optional, Union


class SHAPManager:
    def __init__(
        self,
        random_state: Optional[int] = None,
        cache: bool = True,
    ):
        self.random_state: int = random_state
        self.cache: bool = cache

        if self.cache:
            self.shap_instances: dict[tuple, SHAPInstance] = {}
            self.aggregated_values_cache: dict[tuple, np.ndarray] = {}

    def get_instance(
        self,
        model: BaseModel,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        random_state: Optional[int] = None,
        verbose: Literal[0, 1, 2] = 0,
        cache: Optional[bool] = None,
        **shap_kwargs,
    ) -> SHAPInstance:
        # Create SHAP key, based on only the feature names
        shap_instance_key = create_shap_instance_key(data_manager, split_selection)

        # If cache is not provided, use the class cache
        cache = cache if cache is not None else self.cache

        if cache:
            # Add SHAP instance to the cache & use cache if it exists
            if shap_instance_key not in self.shap_instances:
                if verbose > 0:
                    warnings.warn(
                        f"SHAP instance not found in cache. Calculating SHAP instance for key: {shap_instance_key}"
                    )

                # Create SHAP instance
                self.shap_instances[shap_instance_key] = SHAPInstance(
                    model, data_manager, split_selection, random_state, **shap_kwargs
                )

            return self.shap_instances[shap_instance_key]
        else:
            return SHAPInstance(model, data_manager, split_selection, random_state, **shap_kwargs)

    def get_aggregated_values(
        self,
        shap_instance: SHAPInstance,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        verbose: Literal[0, 1, 2] = 0,
        cache: Optional[bool] = None,
    ) -> np.ndarray:
        # Create a key for the aggregated values
        key = create_aggregation_values_key(
            data_manager=data_manager,
            split_selection=split_selection,
            class_selection=class_selection,
            weights=weights,
            multi_class_aggregation=multi_class_aggregation,
            shap_variance_penalty_factor=shap_variance_penalty_factor,
        )

        # If cache is not provided, use the class cache
        cache = cache if cache is not None else self.cache

        if cache:
            # If the key is not in the aggregated values dictionary, calculate the aggregated values
            if key not in self.aggregated_values_cache:
                if verbose > 0:
                    warnings.warn(f"Aggregation not found in cache. Calculating aggregated values for key: {key}")

                # Calculate the aggregated values
                self.aggregated_values_cache[key] = calculate_aggregated_values(
                    shap_instance=shap_instance,
                    class_selection=class_selection,
                    weights=weights,
                    multi_class_aggregation=multi_class_aggregation,
                    shap_variance_penalty_factor=shap_variance_penalty_factor,
                )

            return self.aggregated_values_cache[key]
        else:
            return calculate_aggregated_values(
                shap_instance=shap_instance,
                class_selection=class_selection,
                weights=weights,
                multi_class_aggregation=multi_class_aggregation,
                shap_variance_penalty_factor=shap_variance_penalty_factor,
            )

    def get_class_values(
        self,
        shap_instance: SHAPInstance,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        class_selection: Optional[Any] = None,
        verbose: Literal[0, 1, 2] = 0,
        cache: bool = True,
    ) -> dict[str, np.ndarray]:
        # Create a key for the aggregated values
        key = create_aggregation_values_key(
            data_manager=data_manager,
            split_selection=split_selection,
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
                self.aggregated_values_cache[key] = calculate_aggregated_values(
                    shap_instance=shap_instance,
                    class_selection=class_selection,
                    weights=None,
                    multi_class_aggregation=None,
                    shap_variance_penalty_factor=0,
                )

            return self.aggregated_values_cache[key]
        else:
            return calculate_aggregated_values(
                shap_instance=shap_instance,
                class_selection=class_selection,
                weights=None,
                multi_class_aggregation=None,
                shap_variance_penalty_factor=0,
            )

    def get_explanation_for_plot(
        self,
        shap_instance: SHAPInstance,
        data_manager: BaseDataManager,
        split_selection: Literal["full", "train", "test"] = "test",
        class_selection: Optional[Any] = None,
        weights: Optional[Dict[Any, float]] = None,
        multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
        shap_variance_penalty_factor: Union[int, float] = 0,
        verbose: Literal[0, 1, 2] = 0,
        cache: Optional[bool] = None,
    ) -> shap.Explanation:
        aggregated_values = self.get_aggregated_values(
            shap_instance,
            data_manager,
            split_selection,
            class_selection,
            weights,
            multi_class_aggregation,
            shap_variance_penalty_factor,
            verbose,
            cache,
        )

        return shap.Explanation(
            values=aggregated_values,
            feature_names=data_manager.column_names,
        )
