import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")


def check_plots_are_generated_correctly(image_path, min_unique_colors=4):
    """
    Check if a plot contains at least a few colors, to ensure that it is not a blank plot.
    But also to ensure it is not a plot with only an axis.

    Args:
        image_path (str): Path to the image file
        min_unique_colors (int): Minimum number of unique colors required

    Returns:
        bool: True if the image has at least min_unique_colors unique colors
    """
    try:
        # Open the image
        img = Image.open(image_path)

        # Convert to RGB if not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get image data as numpy array
        img_data = np.array(img)

        # Reshape to a list of pixels
        pixels = img_data.reshape(-1, 3)

        # Count unique colors (unique rows in the pixel array)
        unique_colors = np.unique(pixels, axis=0)
        num_unique_colors = len(unique_colors)

        # Check if the image has enough unique colors
        return num_unique_colors >= min_unique_colors
    except Exception as e:
        print(f"Error checking image {image_path}: {e}")
        return False


def create_model_with_params(estimator_class, estimator_params, random_state, n_estimators=200):
    """
    Helper function to create a model with the appropriate parameters,
    handling different estimator types that may or may not support n_estimators.

    Args:
        estimator_class: The estimator class to instantiate
        estimator_params: Dictionary of parameters for the estimator
        random_state: Random state for reproducibility
        n_estimators: Number of estimators for tree-based models

    Returns:
        Instantiated model
    """
    try:
        # Try to create with n_estimators if supported
        return estimator_class(
            random_state=random_state,
            n_estimators=n_estimators,
            **{k: v for k, v in estimator_params.items() if k != "n_estimators"},
        )
    except TypeError:
        # Fall back to basic initialization if n_estimators not supported
        return estimator_class(random_state=random_state, **estimator_params)


def get_plots_dir(base_plots_dir, estimator_class, estimators_list):
    """
    Helper function to get the plots directory for a given estimator class.

    Args:
        base_plots_dir: Base directory for plots
        estimator_class: The estimator class
        estimators_list: List of estimator parameters (from ESTIMATORS)

    Returns:
        Path to the plots directory
    """
    estimator_name = next(param.id for param in estimators_list if param.values[0] == estimator_class)
    return os.path.join(base_plots_dir, f"{estimator_name}_plots")


def setup_plot_dirs(save_plots, base_plots_dir, estimators_list):
    """
    Create plot directories only if save_plots is True.

    Args:
        save_plots (bool): Whether to save plots or not
        base_plots_dir (str): Base directory for plots
        estimators_list (list): List of estimator parameters (from ESTIMATORS)
    """
    if save_plots:
        # Create base plots directory if it doesn't exist
        os.makedirs(base_plots_dir, exist_ok=True)

        # Create plots directory for each estimator
        for param in estimators_list:
            estimator_name = param.id
            estimator_plots_dir = os.path.join(base_plots_dir, f"{estimator_name}_plots")
            os.makedirs(estimator_plots_dir, exist_ok=True)
