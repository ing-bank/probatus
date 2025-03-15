import warnings
from typing import Literal, Optional, Union, Iterable

from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator

from probatus.feature_elimination import ShapRFECV
from probatus.utils import Scorer


class EarlyStoppingShapRFECV(ShapRFECV):
    """
    A class that performs Backwards Recursive Feature Elimination using SHAP feature importance with early stopping.

    This class extends ShapRFECV to provide early stopping functionality during model training,
    which is particularly useful for gradient boosting models like LightGBM, XGBoost, and CatBoost.

    Attributes:
        model (BaseEstimator): The model used for feature elimination
        step (Union[int, float]): Number or percentage of features to remove in each round
        min_features_to_select (int): Minimum number of features to keep
        cv (Union[int, BaseCrossValidator, Iterable]): Cross-validation strategy
        scoring (Union[str, Scorer]): Metric used for model evaluation
        n_jobs (int): Number of parallel jobs
        verbose (Literal[0, 1, 2]): Verbosity level
        random_state (Optional[int]): Random seed for reproducibility
        early_stopping_rounds (int): Number of rounds without improvement before stopping
        eval_metric (str): Metric used for early stopping evaluation

    Examples:

    ```python
    from lightgbm import LGBMClassifier
    import pandas as pd
    from probatus.feature_elimination import EarlyStoppingShapRFECV
    from sklearn.datasets import make_classification

    feature_names = [
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
        'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
        'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20']

    # Prepare dataset
    X, y = make_classification(n_samples=200, class_sep=0.05, n_informative=6, n_features=20,
                               random_state=0, n_redundant=10, n_clusters_per_class=1)
    X = pd.DataFrame(X, columns=feature_names)

    # Prepare model
    model = LGBMClassifier(n_estimators=200, max_depth=3)

    # Run feature elimination
    shap_elimination = EarlyStoppingShapRFECV(
        model=model, step=0.2, cv=10, scoring='roc_auc', early_stopping_rounds=10, n_jobs=3)
    report = shap_elimination.fit_compute(X, y)

    # Make plots
    performance_plot = shap_elimination.plot()

    # Get final feature set
    final_features_set = shap_elimination.get_reduced_features_set(num_features=3)
    ```
    """

    def __init__(
        self,
        model: BaseEstimator,
        step: Union[int, float] = 1,
        min_features_to_select: int = 1,
        cv: Optional[Union[int, BaseCrossValidator, Iterable]] = None,
        scoring: Union[str, Scorer] = "roc_auc",
        n_jobs: int = -1,
        verbose: Literal[0, 1, 2] = 0,
        random_state: Optional[int] = None,
        early_stopping_rounds: int = 5,
        eval_metric: str = "auc",
    ) -> None:
        """
        Initialize the EarlyStoppingShapRFECV class.

        Args:
            model (BaseEstimator):
                A model that will be optimized and trained at each round of feature elimination.
                The model must support early stopping of training, which is the case for XGBoost
                and LightGBM. The recommended model is LGBMClassifier, as it handles missing values
                and categorical variables by default.

                This parameter also supports hyperparameter search models that follow the sklearn API
                (GridSearchCV, RandomizedSearchCV, BayesSearchCV). Note that if a hyperparameter search
                model is used, early stopping is only applied during Shapley value estimation, not
                during hyperparameter tuning.

            step (Union[int, float], default=1):
                Number of lowest importance features removed each round:
                - If int: removes that exact number of features each round
                - If float: removes that percentage of remaining features each round (rounded down)

                Using float is recommended for large feature sets as it starts faster and becomes
                more precise as the feature set shrinks.

            min_features_to_select (int, default=1):
                Minimum number of features to keep. This is a stopping criterion for feature elimination.
                If columns_to_keep is specified in the fit method, this parameter may be overridden.

            cv (Optional[Union[int, BaseCrossValidator, Iterable]], default=None):
                Determines the cross-validation splitting strategy. Compatible with sklearn's
                cv parameter. If None, 5-fold cross-validation is used.

            scoring (Union[str, Scorer], default="roc_auc"):
                Metric for model performance evaluation. Can be either:
                - A string matching a predefined sklearn classification scorer name
                - A probatus.utils.Scorer instance for custom metrics

            n_jobs (int, default=-1):
                Number of cores to run in parallel while fitting across folds:
                - None: 1 core (unless in a joblib.parallel_backend context)
                - -1: use all available processors

            verbose (Literal[0, 1, 2], default=0):
                Controls output verbosity:
                - 0: no prints or warnings
                - 1: only important warnings
                - 2: all prints and warnings

            random_state (Optional[int], default=None):
                Random seed for reproducibility. If None, results won't be reproducible
                and different hyperparameters may be tested in each iteration of random search.

            early_stopping_rounds (int, default=5):
                Number of rounds with constant performance after which model fitting stops.
                This is passed to the model's fit method during Shapley value estimation.
                Only supported by certain models like XGBoost and LightGBM.

            eval_metric (str, default="auc"):
                Metric used for scoring fitting rounds and activating early stopping.
                This is passed to the model's fit method during Shapley value estimation.
                Only supported by certain models like XGBoost and LightGBM.
                Note that this is different from the 'scoring' parameter.
        """
        # Issue a deprecation warning as this class functionality is now in ShapRFECV
        warnings.warn(
            "The separate EarlyStoppingShapRFECV class is deprecated "
            "as its functionality is now part of the ShapRFECV class. "
            "Please use ShapRFECV instead of EarlyStoppingShapRFECV.",
            DeprecationWarning,
            stacklevel=2,  # Point to the caller rather than this line
        )

        # Initialize the parent class with all parameters including early stopping parameters
        super().__init__(
            model=model,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric,
        )
