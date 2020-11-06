from probatus.utils import assure_column_names_consistency, assure_pandas_df, shap_calc, assure_list_of_strings,\
    calculate_shap_importance, NotFittedError
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import warnings

class ShapBackwardsFeaturesElimination:

    param_grid = {
        'n_estimators': [25, 50, 100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'num_leaves': [5, 10, 20, 30, 40],
        'min_child_samples': [50, 100, 200, 300, 400, 500],
        'min_child_weight': [1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e4],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1],
        'subsample_freq': [1, 5, 10],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'reg_alpha': [0, 1, 10, 50, 100, 200],
        'reg_lambda': [0, 1, 10, 50, 100, 200]
    }

    def __init__(self, clf='lgbm_balanced', search_schema='random', param_grid=None, n_removed_each_step=2,random_state=42,
                 **opt_kwargs):
        """

        """
        if clf = 'lgbm':
            self.clf = lightgbm.LGBMClassifier()
        elif clf = 'lgbm_balanced':
            self.clf = lightgbm.LGBMClassifier(class_weight='balanced')
        elif isinstance(clf, str) or clf is None:
            raise(ValueError('Unsupported clf, choose one of the following: "lgbm", "lgbm_balanced" or custom model.'))
        else:
            self.clf = clf

        if search_schema == 'random':
            self.opt_class = RandomizedSearchCV
        elif search_schema == 'grid':
            self.opt_class = GridSearchCV
        else:
            raise(ValueError('Unsupported search_schema, choose one of the following: "random", "grid".'))

        if param_grid is not None:
            self.param_grid = param_grid

        self.opt_kwargs = opt_kwargs

        self.n_removed_each_step = n_removed_each_step
        self.random_state = random_state

        self.results_df_columns = ['features_set', 'eliminated_features', 'train_metric_mean', 'train_metric_std',
                              'val_metric_mean', 'val_metric_std'] + [f'param_{param_name}' for param_name in self.param_grid]
        self.results_df = pd.DataFrame([], columns=self.results_df_columns)
        self.fitted = False


    def _check_if_fitted(self):
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))


    def _preprocess_data(self, X):
        # Make sure that X is a pd.DataFrame
        X = assure_pandas_df(X)

        # Warn if missing
        columns_with_missing = [column for column in X.columns if X[column].isnull().values.any()]
        warnings.warn(f'The following variables contain missing values {columns_with_missing}. Make sure to impute'
                      f'missing or apply a model that handles them automatically.')

        # Remove static features, those that have only one value for all samples
        static_features = [i for i in X.columns if len(X[i].unique()) == 1]
        if len(static_features)>0:
            warnings.warn(f'Removing static features f{static_features}.')
            X = X.drop(columns=static_features)

        # Transform Categorical variables into category dtype
        indices_obj_dtype_features = [column[0] for column in enumerate(X.dtypes) if column[1] == 'O']
        obj_dtype_features = list(X.columns[indices_obj_dtype_features])

        # Set categorical features type to category
        if len(obj_dtype_features) > 0:
            warnings.warn(f'Changing dtype of {obj_dtype_features} from "object" to "category". Treating it as '
                          f'categorical variable. Make sure that the model handles categorical variables, or encode '
                          f'them first.')
            for obj_dtype_feature in obj_dtype_features:
                X[obj_dtype_feature] = X[obj_dtype_feature].astype('category')
        return X


    def fit(self, X, y):
        # Set seed for results reproducibility
        np.random.seed(self.random_state)

        self.X = self._preprocess_data(X)
        self.y = y

        remaining_features = self.X.columns.tolist()

        while len(remaining_features) > 0:
            current_X = X[remaining_features]

            # Optimize parameters
            opt_algorithm = self.opt_class(self.clf, self.param_grid, refit=True, **self.opt_kwargs)
            opt_algorithm = opt_algorithm.fit(X, y)

            # Compute SHAP values
            shap_values = shap_calc(opt_algorithm.best_estimator_, current_X)
            shap_importance_df = calculate_shap_importance(shap_values, remaining_features)



