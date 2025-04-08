from probatus.wrapper.data import BaseDataManager


from typing import Any, Dict, List, Literal, Optional, Union


def create_shap_instance_key(
    data_manager: BaseDataManager,
    split_selection: Literal["full", "train", "test"] = "test",
) -> str:
    return f"{split_selection}_{convert_feature_names_list_to_str(data_manager)}"


def create_aggregation_values_key(
    data_manager: BaseDataManager,
    split_selection: Literal["full", "train", "test"] = "test",
    class_selection: Optional[Any] = None,
    weights: Optional[Dict[Any, float]] = None,
    multi_class_aggregation: Optional[Literal["max_abs", "mean", "mean_abs"]] = None,
    shap_variance_penalty_factor: Union[int, float] = 0,
) -> tuple[
    str,  # split_selection
    List[str],  # Class names
    Union[str, int],  # Class selection
    str,  # Weights (dict formatted as a ordered string)
    Union[str, Literal["mean", "mean_abs", "max_abs"]],  # Aggregation method
    Union[str, float, int],  # Penalty factor
]:
    # Convert complex objects to strings
    class_names_str = convert_feature_names_list_to_str(data_manager)
    weights_str = convert_weights_dict_to_str(weights)

    # Create a key for the aggregated values
    return (
        split_selection,
        class_names_str,
        class_selection if class_selection else "_",
        weights_str if weights_str else "_",
        multi_class_aggregation if multi_class_aggregation else "_",
        shap_variance_penalty_factor if shap_variance_penalty_factor else 0,
    )


def convert_feature_names_list_to_str(data_manager: BaseDataManager) -> str:
    return "_-_".join(data_manager.column_names)


def convert_weights_dict_to_str(weights: dict[Any, float]) -> str:
    # First order the dictionary by key
    sorted_dictionary = dict(sorted(weights.items()))

    # Then convert to a string
    return "".join(f"{k}:{v}" for k, v in sorted_dictionary.items())
