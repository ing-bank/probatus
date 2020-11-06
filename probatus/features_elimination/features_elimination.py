from probatus.utils import assure_pandas_df, shap_calc, assure_list_of_strings, calculate_shap_importance, \
    NotFittedError
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import warnings

class ShapBackwardsFeaturesElimination:
    """

    Example:
    ```python
    from probatus.features_elimination import ShapBackwardsFeaturesElimination
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    feature_names = ['f1_categorical', 'f2_missing', 'f3_static', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15']

    # Prepare two samples
    X, y = make_classification(n_samples=1000, n_features=15, random_state=0, n_redundant=10)
    X = pd.DataFrame(X, columns=feature_names)
    X['f1_categorical'] = X['f1_categorical'].apply(lambda x: str(np.round(x*10)))
    X['f2_missing'] = X['f2_missing'].apply(lambda x: x if np.random.rand()<0.8 else np.nan)
    X['f3_static'] = 0

    # Run feature elimination
    shap_elimination = ShapBackwardsFeaturesElimination(
        clf='lgbm_balanced', search_schema='random',
        n_removed_each_step=0.2, cv=10, n_iter=50, scoring='roc_auc', n_jobs=3)
    report = shap_elimination.fit_compute(X, y)

    # Make plots
    shap_elimination.plot('performance')
    shap_elimination.plot('parameter', param_names=['n_estimators', 'learning_rate'])
    ```
    """


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

    def __init__(self, clf='lgbm_balanced', search_schema='random', param_grid=None, n_removed_each_step='',random_state=42,
                 **opt_kwargs):
        """

        """
        if clf == 'lgbm':
            self.clf = lightgbm.LGBMClassifier()
        elif clf == 'lgbm_balanced':
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

        self.report_df = pd.DataFrame([])
        self.fitted = False


    def _check_if_fitted(self):
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))


    def _preprocess_data(self, X):
        # Make sure that X is a pd.DataFrame
        X = assure_pandas_df(X)

        # Remove static features, those that have only one value for all samples
        static_features = [i for i in X.columns if len(X[i].unique()) == 1]
        if len(static_features)>0:
            warnings.warn(f'Removing static features f{static_features}.')
            X = X.drop(columns=static_features)

        # Warn if missing
        columns_with_missing = [column for column in X.columns if X[column].isnull().values.any()]
        if len(columns_with_missing) > 0:
            warnings.warn(f'The following variables contain missing values {columns_with_missing}. Make sure to impute'
                          f'missing or apply a model that handles them automatically.')

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


    def _get_current_features_to_remove(self, shap_importance_df):
        # If the n_removed_each_step is an int remove n features.
        if isinstance(self.n_removed_each_step, int) and self.n_removed_each_step > 0:
            if self.n_removed_each_step > shap_importance_df.shape[0]:
                features_to_remove = shap_importance_df.index.tolist()
            else:
                features_to_remove = shap_importance_df.iloc[-self.n_removed_each_step:].index.tolist()
        # If the n_removed_each_step is a float remove n * number features that are left, rounded up.
        elif isinstance(self.n_removed_each_step, float) and self.n_removed_each_step > 0:
            num_features_to_remove = int(np.ceil(shap_importance_df.shape[0] * self.n_removed_each_step))
            features_to_remove = shap_importance_df.iloc[-num_features_to_remove:].index.tolist()
        else:
            raise(ValueError(f"The current value of n_removed_each_step = {self.n_removed_each_step} is not allowed. "
                             f"It needs to be a positive integer or positive float."))
        return features_to_remove


    def _report_current_results(self, round_number, current_features_set, features_to_remove, opt_algorithm):
        current_results = {
            'num_features': len(current_features_set),
            'features_set': None,
            'eliminated_features':  None,
            'train_metric_mean': np.round(opt_algorithm.cv_results_['mean_train_score'][opt_algorithm.best_index_], 3),
            'train_metric_std': np.round(opt_algorithm.cv_results_['std_train_score'][opt_algorithm.best_index_], 3),
            'val_metric_mean': np.round(opt_algorithm.cv_results_['mean_test_score'][opt_algorithm.best_index_], 3),
            'val_metric_std': np.round(opt_algorithm.cv_results_['std_test_score'][opt_algorithm.best_index_], 3),
        }

        for param_name, param_value in opt_algorithm.best_params_.items():
            current_results[f'param_{param_name}'] = param_value

        current_row = pd.DataFrame(current_results, index=[round_number])
        current_row['features_set'] = [current_features_set]
        current_row['eliminated_features'] = [features_to_remove]

        self.report_df = pd.concat([self.report_df, current_row], axis=0)


    def fit(self, X, y):
        # Set seed for results reproducibility
        np.random.seed(self.random_state)

        self.X = self._preprocess_data(X)
        self.y = y

        remaining_features = self.X.columns.tolist()
        round_number = 0

        while len(remaining_features) > 0:
            round_number += 1
            # Set seed at each iteration to make sure same parameters sets are tested.
            np.random.seed(self.random_state)

            # Get current dataset info
            current_features_set = remaining_features
            current_X = self.X[current_features_set]

            # Optimize parameters
            opt_algorithm = self.opt_class(estimator=self.clf, param_distributions=self.param_grid, refit=True,
                                           return_train_score=True, **self.opt_kwargs)
            opt_algorithm = opt_algorithm.fit(current_X, y)

            # Compute SHAP values
            shap_values = shap_calc(opt_algorithm.best_estimator_, current_X, suppress_warnings=True)
            shap_importance_df = calculate_shap_importance(shap_values, remaining_features)

            # Get features to remove
            features_to_remove = self._get_current_features_to_remove(shap_importance_df)
            remaining_features = list(set(current_features_set) - set(features_to_remove))

            # Report results
            self._report_current_results(round_number=round_number, current_features_set=current_features_set,
                                         features_to_remove=features_to_remove, opt_algorithm=opt_algorithm)
            print(f'Round: {round_number}, Current number of features: {len(current_features_set)}, '
                  f'Current performance: Train {self.report_df.loc[round_number]["train_metric_mean"]} '
                  f'+/- {self.report_df.loc[round_number]["train_metric_std"]}, Test '
                  f'{self.report_df.loc[round_number]["train_metric_mean"]} '
                  f'+/- {self.report_df.loc[round_number]["train_metric_std"]}. \n'
                  f'Removed features at the end of the round: {features_to_remove}, '
                  f'Num of features left: {len(remaining_features)}.')
        self.fitted = True


    def compute(self):
        """
        Checks if fit() method has been run and computes the DataFrame with results of features elimintation for each
         round.

        Returns:
            (pd.DataFrame): DataFrame with results of features elimination for each round.
        """
        self._check_if_fitted()

        return self.report_df


    def fit_compute(self, X, y):
        self.fit(X, y)
        return self.compute()


    def plot(self, plot_type='performance', param_names=None, **figure_kwargs):
        if plot_type == 'performance':
            plt.figure(**figure_kwargs)

            plt.plot(self.report_df['num_features'], self.report_df['train_metric_mean'], label='Train Score')
            plt.fill_between(pd.to_numeric(self.report_df.num_features, errors='coerce'),
                             self.report_df['train_metric_mean'] - self.report_df['train_metric_std'],
                             self.report_df['train_metric_mean'] + self.report_df['train_metric_std'], alpha=.3)

            plt.plot(self.report_df['num_features'], self.report_df['val_metric_mean'], label='Validation Score')
            plt.fill_between(pd.to_numeric(self.report_df.num_features, errors='coerce'),
                             self.report_df['val_metric_mean'] - self.report_df['val_metric_std'],
                             self.report_df['val_metric_mean'] + self.report_df['val_metric_std'], alpha=.3)

            plt.xlabel('Number of features')
            plt.ylabel('Performance')
            plt.title('Feature selection using SHAP')
            plt.legend(loc="lower right")
            ax = plt.gca()
            ax.invert_xaxis()
            plt.show()

        elif plot_type == 'parameter':
            param_names = assure_list_of_strings(param_names, 'target_columns')
            ax = []
            for param_name in param_names:
                if param_name not in self.param_grid.keys():
                    raise(ValueError(f'Param {param_name} not in parameter search grid.'))
                plt.figure(**figure_kwargs)

                plt.plot(self.report_df['num_features'], self.report_df[f'param_{param_name}'], label=f'{param_name} optimized value')
                plt.xlabel('Number of features')
                plt.ylabel(f'Optimizal {param_name} value')
                plt.title(f'Optimization of {param_name} for different numbers of features')
                plt.legend(loc="lower right")
                current_ax = plt.gca()
                current_ax.invert_xaxis()
                ax.append(current_ax)
                plt.show()
        else:
            raise(ValueError('Wrong value of plot_type. Select from "performance" or "parameter"'))

        if isinstance(ax, list) and len(ax) == 1:
            ax = ax[0]
        return ax

