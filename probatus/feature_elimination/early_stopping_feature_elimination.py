import warnings

from probatus.utils import (
    shap_calc,
)
from probatus.feature_elimination import ShapRFECV


class EarlyStoppingShapRFECV(ShapRFECV):
    """
    This class performs Backwards Recursive Feature Elimination, using SHAP feature importance.

    This is a child of ShapRFECV which allows early stopping of the training step, this class is compatible with
        LightGBM, XGBoost and CatBoost models. If you are not using early stopping, you should use the parent class,
        ShapRFECV, instead of EarlyStoppingShapRFECV.

    [Early stopping](https://en.wikipedia.org/wiki/Early_stopping) is a type of
        regularization technique in which the model is trained until the scoring metric, measured on a validation set,
        stops improving after a number of early_stopping_rounds. In boosted tree models, this technique can increase
        the training speed, by skipping the training of trees that do not improve the scoring metric any further,
        which is particularly useful when the training dataset is large.

    Note that if the regressor or classifier is a hyperparameter search model is used, the early stopping parameter is passed only
        to the fit method of the model duiring the Shapley values estimation step, and not for the hyperparameter
        search step.
        Early stopping can be seen as a type of regularization of the optimal number of trees. Therefore you can use
        it directly with a LightGBM or XGBoost model, as an alternative to a hyperparameter search model.

    At each round, for a
        given feature set, starting from all available features, the following steps are applied:

    1. (Optional) Tune the hyperparameters of the model using sklearn compatible search CV e.g.
        [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html),
        [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomized#sklearn.model_selection.RandomizedSearchCV), or
        [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html).
        Note that during this step the model does not use early stopping.
    2. Apply Cross-validation (CV) to estimate the SHAP feature importance on the provided dataset. In each CV
        iteration, the model is fitted on the train folds, and applied on the validation fold to estimate
        SHAP feature importance. The model is trained until the scoring metric eval_metric, measured on the
        validation fold, stops improving after a number of early_stopping_rounds.
    3. Remove `step` lowest SHAP importance features from the dataset.

    At the end of the process, the user can plot the performance of the model for each iteration, and select the
        optimal number of features and the features set.

    We recommend using [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
        because by default it handles missing values and categorical features. In case of other models, make sure to
        handle these issues for your dataset and consider impact it might have on features importance.


    Example:
    ```python
    from lightgbm import LGBMClassifier
    import pandas as pd
    from probatus.feature_elimination import EarlyStoppingShapRFECV
    from sklearn.datasets import make_classification

    feature_names = [
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
        'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
        'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20']

    # Prepare two samples
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
    <img src="../img/earlystoppingshaprfecv.png" width="500" />

    """  # noqa

    def __init__(
        self,
        model,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        random_state=None,
        early_stopping_rounds=5,
        eval_metric="auc",
    ):
        """
        This method initializes the class.

        Args:
            model (sklearn compatible classifier or regressor, sklearn compatible search CV e.g. GridSearchCV, RandomizedSearchCV or BayesSearchCV):
                A model that will be optimized and trained at each round of features elimination. The model must
                support early stopping of training, which is the case for XGBoost and LightGBM, for example. The
                recommended model is [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
                because it by default handles the missing values and categorical variables. This parameter also supports
                any hyperparameter search schema that is consistent with the sklearn API e.g.
                [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
                [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
                or [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html#skopt.BayesSearchCV).
                Note that if a hyperparemeter search model is used, the hyperparameters are tuned without early
                stopping. Early stopping is applied only during the Shapley values estimation for feature
                elimination. We recommend simply passing the model without hyperparameter optimization, or using
                ShapRFECV without early stopping.


            step (int or float, optional):
                Number of lowest importance features removed each round. If it is an int, then each round such number of
                features is discarded. If float, such percentage of remaining features (rounded down) is removed each
                iteration. It is recommended to use float, since it is faster for a large number of features, and slows
                down and becomes more precise towards less features. Note: the last round may remove fewer features in
                order to reach min_features_to_select.
                If columns_to_keep parameter is specified in the fit method, step is the number of features to remove after
                keeping those columns.

            min_features_to_select (int, optional):
                Minimum number of features to be kept. This is a stopping criterion of the feature elimination. By
                default the process stops when one feature is left. If columns_to_keep is specified in the fit method,
                it may override this parameter to the maximum between length of columns_to_keep the two.

            cv (int, cross-validation generator or an iterable, optional):
                Determines the cross-validation splitting strategy. Compatible with sklearn
                [cv parameter](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).
                If None, then cv of 5 is used.

            scoring (string or probatus.utils.Scorer, optional):
                Metric for which the model performance is calculated. It can be either a metric name  aligned with predefined
                [classification scorers names in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html).
                Another option is using probatus.utils.Scorer to define a custom metric.

            n_jobs (int, optional):
                Number of cores to run in parallel while fitting across folds. None means 1 unless in a
                `joblib.parallel_backend` context. -1 means using all processors.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - only most important warnings
                - 2 - shows all prints and all warnings.

            random_state (int, optional):
                Random state set at each round of feature elimination. If it is None, the results will not be
                reproducible and in random search at each iteration a different hyperparameters might be tested. For
                reproducible results set it to integer.

            early_stopping_rounds (int, optional):
                Number of rounds with constant performance after which the model fitting stops. This is passed to the
                fit method of the model for Shapley values estimation, but not for hyperparameter search. Only
                supported by some models, such as XGBoost and LightGBM.

            eval_metric (str, optional):
                Metric for scoring fitting rounds and activating early stopping. This is passed to the
                fit method of the model for Shapley values estimation, but not for hyperparameter search. Only
                supported by some models, such as [XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
                and [LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters).
                Note that `eval_metric` is an argument of the model's fit method and it is different from `scoring`.
        """  # noqa
        super().__init__(
            model,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

        if self.search_model and self.verbose > 0:
            warnings.warn(
                "Early stopping will be used only during Shapley value"
                " estimation step, and not for hyperparameter"
                " optimization."
            )

        if not isinstance(early_stopping_rounds, int) or early_stopping_rounds <= 0:
            raise ValueError(
                f"The current value of early_stopping_rounds ="
                f" {early_stopping_rounds} is not allowed."
                f" It needs to be a positive integer."
            )

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

    def _get_fit_params_lightGBM(
        self, X_train, y_train, X_val, y_val, sample_weight=None, train_index=None, val_index=None
    ):
        """Get the fit parameters for for a LightGBM Model.

        Args:

            X_train (pd.DataFrame):
                Train Dataset used in CV.

            y_train (pd.Series):
                Train labels for X.

            X_val (pd.DataFrame):
                Validation Dataset used in CV.

            y_val (pd.Series):
                Validation labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

        Raises:
            ValueError: if the model is not supported.

        Returns:
            dict: fit parameters
        """
        from lightgbm import early_stopping, log_evaluation

        fit_params = {
            "X": X_train,
            "y": y_train,
            "eval_set": [(X_val, y_val)],
            "eval_metric": self.eval_metric,
            "callbacks": [
                early_stopping(self.early_stopping_rounds, first_metric_only=True),
                log_evaluation(1 if self.verbose >= 2 else 0),
            ],
        }

        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_index]
            fit_params["eval_sample_weight"] = [sample_weight.iloc[val_index]]

        return fit_params

    def _get_fit_params_XGBoost(
        self, X_train, y_train, X_val, y_val, sample_weight=None, train_index=None, val_index=None
    ):
        """Get the fit parameters for for a XGBoost Model.

        Args:

            X_train (pd.DataFrame):
                Train Dataset used in CV.

            y_train (pd.Series):
                Train labels for X.

            X_val (pd.DataFrame):
                Validation Dataset used in CV.

            y_val (pd.Series):
                Validation labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

        Raises:
            ValueError: if the model is not supported.

        Returns:
            dict: fit parameters
        """
        fit_params = {
            "X": X_train,
            "y": y_train,
            "eval_set": [(X_val, y_val)],
        }
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_index]
            fit_params["eval_sample_weight"] = [sample_weight.iloc[val_index]]

        return fit_params

    def _get_fit_params_CatBoost(
        self, X_train, y_train, X_val, y_val, sample_weight=None, train_index=None, val_index=None
    ):
        """Get the fit parameters for for a CatBoost Model.

        Args:

            X_train (pd.DataFrame):
                Train Dataset used in CV.

            y_train (pd.Series):
                Train labels for X.

            X_val (pd.DataFrame):
                Validation Dataset used in CV.

            y_val (pd.Series):
                Validation labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

        Raises:
            ValueError: if the model is not supported.

        Returns:
            dict: fit parameters
        """
        from catboost import Pool

        cat_features = [col for col in X_train.select_dtypes(include=["category"]).columns]
        fit_params = {
            "X": Pool(X_train, y_train, cat_features=cat_features),
            "eval_set": Pool(X_val, y_val, cat_features=cat_features),
            # Evaluation metric should be passed during initialization
        }
        if sample_weight is not None:
            fit_params["X"].set_weight(sample_weight.iloc[train_index])
            fit_params["eval_set"].set_weight(sample_weight.iloc[val_index])

        return fit_params

    def _get_fit_params(
        self, model, X_train, y_train, X_val, y_val, sample_weight=None, train_index=None, val_index=None
    ):
        """Get the fit parameters for the specified classifier or regressor.

        Args:
            model (classifier or regressor):
                Model to be fitted on the train folds.

            X_train (pd.DataFrame):
                Train Dataset used in CV.

            y_train (pd.Series):
                Train labels for X.

            X_val (pd.DataFrame):
                Validation Dataset used in CV.

            y_val (pd.Series):
                Validation labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

        Raises:
            ValueError: if the model is not supported.

        Returns:
            dict: fit parameters
        """
        # The lightgbm and xgboost imports are temporarily placed here, until the tests on
        # macOS have been fixed.

        try:
            from lightgbm import LGBMModel

            if isinstance(model, LGBMModel):
                return self._get_fit_params_lightGBM(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    sample_weight=sample_weight,
                    train_index=train_index,
                    val_index=val_index,
                )
        except ImportError:
            pass

        try:
            from xgboost.sklearn import XGBModel

            if isinstance(model, XGBModel):
                return self._get_fit_params_XGBoost(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    sample_weight=sample_weight,
                    train_index=train_index,
                    val_index=val_index,
                )
        except ImportError:
            pass

        try:
            from catboost import CatBoost

            if isinstance(model, CatBoost):
                return self._get_fit_params_CatBoost(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    sample_weight=sample_weight,
                    train_index=train_index,
                    val_index=val_index,
                )
        except ImportError:
            pass

        raise ValueError("Model type not supported")

    def _get_feature_shap_values_per_fold(
        self,
        X,
        y,
        model,
        train_index,
        val_index,
        sample_weight=None,
        **shap_kwargs,
    ):
        """
        This function calculates the shap values on validation set, and Train and Val score.

        Args:
            X (pd.DataFrame):
                Dataset used in CV.

            y (pd.Series):
                Labels for X.

            sample_weight (pd.Series, np.ndarray, list, optional):
                array-like of shape (n_samples,) - only use if the model you're using supports
                sample weighting (check the corresponding scikit-learn documentation).
                Array of weights that are assigned to individual samples.
                Note that they're only used for fitting of  the model, not during evaluation of metrics.
                If not provided, then each sample is given unit weight.

            model:
                Classifier or regressor to be fitted on the train folds.

            train_index (np.array):
                Positions of train folds samples.

            val_index (np.array):
                Positions of validation fold samples.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.
        Returns:
            (np.array, float, float):
                Tuple with the results: Shap Values on validation fold, train score, validation score.
        """
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        fit_params = self._get_fit_params(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            train_index=train_index,
            val_index=val_index,
        )

        # Due to deprecation issues (compatibility with Sklearn) set some params
        # like below, instead of through fit().
        try:
            from xgboost.sklearn import XGBModel

            if isinstance(model, XGBModel):
                model.set_params(eval_metric=self.eval_metric, early_stopping_rounds=self.early_stopping_rounds)
        except ImportError:
            pass

        try:
            from catboost import CatBoost

            if isinstance(model, CatBoost):
                model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        except ImportError:
            pass

        # Train the model
        model = model.fit(**fit_params)

        # Score the model
        score_train = self.scorer.score(model, X_train, y_train)
        score_val = self.scorer.score(model, X_val, y_val)

        # Compute SHAP values
        shap_values = shap_calc(model, X_val, verbose=self.verbose, random_state=self.random_state, **shap_kwargs)
        return shap_values, score_train, score_val
