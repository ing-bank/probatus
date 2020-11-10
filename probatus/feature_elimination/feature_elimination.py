from probatus.utils import assure_pandas_df, shap_calc, assure_list_of_strings, calculate_shap_importance, \
    NotFittedError
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import warnings

class ShapRFECV:
    """
    This class performs Backwards Recursive Feature Elimination, using SHAP features importance. At each round, for a given
     features set, starting from all available features, a model is optimized (e.g. using RandomSearchCV) and trained.
     At the end of each round, the n lowest SHAP feature importance features are removed and the model results are
     stored. The user can plot the performance of the model for each round, and select the optimal number of features
     and the features set.

    The functionality is similar to [sklearn.feature_selection.RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html),
     yet, it removes the lowest importance features based on SHAP features importance and optimizes the hyperparameters
     of the model at each round.

    We recommend using LightGBM model, because by default it handles missing values and categorical features. In case
     of other models, make sure to handle these issues for your dataset and consider impact it might have on features
     importance.

    Example:
    ```python
    from probatus.feature_elimination import ShapRFECV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    import lightgbm

    feature_names = ['f1_categorical', 'f2_missing', 'f3_static', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15']

    # Prepare two samples
    X, y = make_classification(n_samples=1000, n_features=15, random_state=0, n_redundant=10)
    X = pd.DataFrame(X, columns=feature_names)
    X['f1_categorical'] = X['f1_categorical'].apply(lambda x: str(np.round(x*10)))
    X['f2_missing'] = X['f2_missing'].apply(lambda x: x if np.random.rand()<0.8 else np.nan)
    X['f3_static'] = 0

    # Prepare model and parameter search space
    clf = lightgbm.LGBMClassifier(max_depth=5, class_weight='balanced')
    param_grid = {
        'n_estimators': [5, 7, 10],
        'num_leaves': [3, 5, 7],
    }

        # Run feature elimination
    shap_elimination = ShapRFECV(
        clf=clf, search_space=param_grid, search_schema='grid',
        step=0.2, cv=20, scoring='roc_auc', n_jobs=3, random_state=42)
    report = shap_elimination.fit_compute(X, y)

    # Make plots
    shap_elimination.plot('performance')
    shap_elimination.plot('parameter', param_names=['n_estimators', 'num_leaves'])

    # Get final features set
    final_features_set = shap_elimination.get_reduced_features_set(num_features=2)
    ```
    """

    def __init__(self, clf, search_space, search_schema='random', step=1, min_features_to_select=1,
                 random_state=None, **search_kwargs):
        """
        This method initializes the class:

        Args:
            clf (binary classifier): A model that will be optimized and trained at each round of features
             elimination. The recommended model is LightGBM, because it by default handles the missing values and
             categorical variables.

            search_space (dict of sklearn.ParamGrid): Parameter search space, which will be explored during the
             hyperparameter search. In case `grid` search_schema, it is passed to GridSearchCV as `param_grid`, in case
             of `random` search_schema, then this value is passed to RandomSearchCV as `param_distributions` parameter.

            search_schema (Optional, str): The hyperparameter search algorithm that should be used to optimize the model.
              It can be one of the following:

                - `random`: [RandomSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
                  which randomly selects hyperparameters from the prowided param_grid, and performs optimization using
                  Cross-Validation. It is recommended option, when you optimize a large number of hyperparameters.
                - `grid`: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
                  which searches through all permutations of hyperparameters values from param_grid, and performs
                  optimization using Cross-Validation. It is recommended option, for low number of hyperparameters.

            step (Optional, int or float): Number of lowest importance features removed each round. If it
             is an int, then each round such number of features is discarded. If float, such percentage of remaining
             features (rounded down) is removed each iteration. It is recommended to use float, since it is faster for a
             large number of features, and slows down and becomes more precise towards less features. Note: the last
             round may remove fewer features in order to reach min_features_to_select.

            min_features_to_select (Optional, unt): Minimum number of features to be kept. This is a stopping criterion
             of the feature elimination. By default the process stops when one feature is left.

            random_state (Optional, int): Random state set at each round of feature elimination. If it is None, the
             results will not be reproducible and in random search at each iteration a different hyperparameters might
             be tested. For reproducible results set it to integer.

            **search_kwargs: The keywords arguments passed to a given search schema, during initialization. Please refer
             to the parameters of a given search schema.
        """
        self.clf = clf

        if search_schema == 'random':
            self.search_class = RandomizedSearchCV
        elif search_schema == 'grid':
            self.search_class = GridSearchCV
        else:
            raise(ValueError('Unsupported search_schema, choose one of the following: "random", "grid".'))

        self.search_space = search_space

        self.search_kwargs = search_kwargs

        if (isinstance(step, int) or isinstance(step, float)) and \
                step > 0:
            self.step = step
        else:
            raise (ValueError(f"The current value of step = {step} is not allowed. "
                              f"It needs to be a positive integer or positive float."))

        if isinstance(min_features_to_select, int) and min_features_to_select>0:
            self.min_features_to_select=min_features_to_select
        else:
            raise (ValueError(f"The current value of min_features_to_select = {min_features_to_select} is not allowed. "
                              f"It needs to be a positive integer."))

        self.random_state = random_state

        self.report_df = pd.DataFrame([])
        self.fitted = False


    def _check_if_fitted(self):
        """
        Checks if object has been fitted. If not, NotFittedError is raised.
        """
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))

    @staticmethod
    def _preprocess_data(X):
        """
        Does basic preprocessing of the data: Removal of static features, Warns which features have missing variables,
        and transform object dtype features to category type, such that LightGBM handles them by default.

        Args:
            X (pd.DataFrame): Provided dataset.

        Returns:
            (pd.DataFrame): Preprocessed dataset.
        """
        # Make sure that X is a pd.DataFrame
        X = assure_pandas_df(X)

        # Remove static features, those that have only one value for all samples
        static_features = [i for i in X.columns if len(X[i].unique()) == 1]
        if len(static_features)>0:
            warnings.warn(f'Removing static features {static_features}.')
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
        """
        Implements the logic used to determine which features to remove. If step is a positive integer,
        at each round step lowest SHAP importance features are selected. If it is a float, such percentage
        of remaining features (rounded up) is removed each iteration. It is recommended to use float, since it is faster
        for a large set of features, and slows down and becomes more precise towards less features.

        Args:
            shap_importance_df (pd.DataFrame): DataFrame presenting SHAP importance of remaining features.

        Returns:
            (list): List of features to be removed at a given round.
        """

        # If the step is an int remove n features.
        if isinstance(self.step, int):
            num_features_to_remove = self._calculate_number_of_features_to_remove(
                current_num_of_features=shap_importance_df.shape[0],
                num_features_to_remove=self.step,
                min_num_features_to_keep=self.min_features_to_select
            )
        # If the step is a float remove n * number features that are left, rounded down
        elif isinstance(self.step, float):
            current_step = int(np.floor(shap_importance_df.shape[0] * self.step))
            # The step after rounding down should be at least 1
            if current_step < 1:
                current_step = 1

            num_features_to_remove = self._calculate_number_of_features_to_remove(
                current_num_of_features=shap_importance_df.shape[0],
                num_features_to_remove=current_step,
                min_num_features_to_keep=self.min_features_to_select
            )

        if num_features_to_remove == 0:
            return []
        else:

            return shap_importance_df.iloc[-num_features_to_remove:].index.tolist()


    @staticmethod
    def _calculate_number_of_features_to_remove(current_num_of_features, num_features_to_remove,
                                                min_num_features_to_keep):
        """
        Calculates the number of features to be removed, and makes sure that after removal at least
         min_num_features_to_keep are kept

         Args:
            current_num_of_features (int): Current number of features in the data.
            num_features_to_remove (int): Number of features to be removed at this stage.
            min_num_features_to_keep (int): Minimum number of features to be left after removal.

        Returns:
            (int) Number of features to be removed.
        """
        num_features_after_removal = current_num_of_features - num_features_to_remove
        if num_features_after_removal >= min_num_features_to_keep:
            num_to_remove = num_features_to_remove
        else:
            # take all available features minus number of them that should stay
            num_to_remove = current_num_of_features - min_num_features_to_keep
        return num_to_remove


    def _report_current_results(self, round_number, current_features_set, features_to_remove, search):
        """
        This function adds the results from a current iteration to the report.

        Args:
            round_number (int): Current number of the round.
            current_features_set (list of str): Current list of features.
            features_to_remove (list of str): List of features to be removed at the end of this iteration.
            search (sklearn.GridSearchCV or sklearn.RandomSearchCV): The fitted hyperparameter search object, containing
             results of the optimization.
        """
        current_results = {
            'num_features': len(current_features_set),
            'features_set': None,
            'eliminated_features':  None,
            'train_metric_mean': np.round(search.cv_results_['mean_train_score'][search.best_index_], 3),
            'train_metric_std': np.round(search.cv_results_['std_train_score'][search.best_index_], 3),
            'val_metric_mean': np.round(search.cv_results_['mean_test_score'][search.best_index_], 3),
            'val_metric_std': np.round(search.cv_results_['std_test_score'][search.best_index_], 3),
        }

        for param_name, param_value in search.best_params_.items():
            current_results[f'param_{param_name}'] = param_value

        current_row = pd.DataFrame(current_results, index=[round_number])
        current_row['features_set'] = [current_features_set]
        current_row['eliminated_features'] = [features_to_remove]

        self.report_df = pd.concat([self.report_df, current_row], axis=0)


    def fit(self, X, y):
        """
        Fits the object with the provided data. The algorithm starts with the entire dataset, and then sequentially
         eliminates features. At each step, it optimizes hyperparameters of the model, computes SHAP features importance
         and removes the lowest importance features.

        Args:
            X (pd.DataFrame): Provided dataset.
            y (pd.Series): Binary labels for X.
        """
        # Set seed for results reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.X = self._preprocess_data(X)
        self.y = y

        remaining_features = current_features_set = self.X.columns.tolist()
        round_number = 0

        while len(current_features_set) > self.min_features_to_select:
            round_number += 1

            # Get current dataset info
            current_features_set = remaining_features
            current_X = self.X[current_features_set]

            # Optimize parameters
            # Set seed for results reproducibility
            if self.random_state is not None:
                np.random.seed(self.random_state)

            search = self.search_class(self.clf, self.search_space, refit=True,
                                       return_train_score=True, **self.search_kwargs)
            search = search.fit(current_X, y)

            # Compute SHAP values
            shap_values = shap_calc(search.best_estimator_, current_X, suppress_warnings=True)
            shap_importance_df = calculate_shap_importance(shap_values, remaining_features)

            # Get features to remove
            features_to_remove = self._get_current_features_to_remove(shap_importance_df)
            remaining_features = list(set(current_features_set) - set(features_to_remove))

            # Report results
            self._report_current_results(round_number=round_number, current_features_set=current_features_set,
                                         features_to_remove=features_to_remove, search=search)

            print(f'Round: {round_number}, Current number of features: {len(current_features_set)}, '
                  f'Current performance: Train {self.report_df.loc[round_number]["train_metric_mean"]} '
                  f'+/- {self.report_df.loc[round_number]["train_metric_std"]}, CV Validation '
                  f'{self.report_df.loc[round_number]["val_metric_mean"]} '
                  f'+/- {self.report_df.loc[round_number]["val_metric_std"]}. \n'
                  f'Num of features left: {len(remaining_features)}. '
                  f'Removed features at the end of the round: {features_to_remove}')
        self.fitted = True


    def compute(self):
        """
        Checks if fit() method has been run and computes the DataFrame with results of feature elimintation for each
         round.

        Returns:
            (pd.DataFrame): DataFrame with results of feature elimination for each round.
        """
        self._check_if_fitted()

        return self.report_df


    def fit_compute(self, X, y):
        """
        Fits the object and computes the report. The algorithm starts with the entire dataset, and then sequentially
         eliminates features. At each step, it optimizes hyperparameters of the model, computes SHAP features importance
         and removes the lowest importance features. At the end, the report containing results from each iteration is
         computed and returned to the user.

        Args:
            X (pd.DataFrame): Provided dataset.
            y (pd.Series): Binary labels for X.

        Returns:
            (pd.DataFrame): DataFrame containing results of feature elimination from each iteration.
        """

        self.fit(X, y)
        return self.compute()


    def get_reduced_features_set(self, num_features):
        """
        Gets the features set after the feature elimination process, for a given number of features.

        Args:
            num_features (int): Number of features in the reduced features set.

        Returns:
            (list of str): Reduced features set.
        """
        self._check_if_fitted()

        if num_features not in self.report_df.num_features.tolist():
            raise(ValueError(f'The provided number of features has not been achieved at any stage of the process. '
                             f'You can select one of the following: {self.report_df.num_features.tolist()}'))
        else:
            return self.report_df[self.report_df.num_features == num_features]['features_set'].values[0]


    def plot(self, plot_type='performance', param_names=None, show=True, **figure_kwargs):
        """
        Generates plots that allow to analyse the results.

        Args:
            plot_type (Optional, str): String indicating the plot type:

                - `performance`: Performance of the optimized model at each iteration. This plot allows to select the
                 optimal features set.
                - `parameter`: Plots the optimized hyperparameter's values at each iteration. This plot allows to
                 analyse stability of parameters for different features set. In case large variability of optimal
                 hyperparameters values is seen, consider reducing the search space.

            param_names (Optional, str, list of str): Name or names of parameters that will be plotted in case of
            `plot_type="parameter"`

            show (Optional, bool): If True, the plots are showed to the user, otherwise they are not shown.

            **figure_kwargs: Keyword arguments that are passed to the plt.figure, at its initialization.

        Returns:
            (plt.axis or list of plt.axis) Axis containing the target plot, or list of such axes.
        """
        x_ticks = list(reversed(self.report_df['num_features'].tolist()))

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
            plt.title('Backwards Feature Elimination using SHAP & CV')
            plt.legend(loc="lower left")
            ax = plt.gca()
            ax.invert_xaxis()
            ax.set_xticks(x_ticks)
            if show:
                plt.show()
            else:
                plt.close()

        elif plot_type == 'parameter':
            param_names = assure_list_of_strings(param_names, 'target_columns')
            ax = []
            for param_name in param_names:
                plt.figure(**figure_kwargs)

                plt.plot(self.report_df['num_features'], self.report_df[f'param_{param_name}'],
                         label=f'{param_name} optimized value')
                plt.xlabel('Number of features')
                plt.ylabel(f'Optimizal {param_name} value')
                plt.title(f'Optimization of {param_name} for different numbers of features')
                plt.legend(loc="lower left")
                current_ax = plt.gca()
                current_ax.invert_xaxis()
                current_ax.set_xticks(x_ticks)
                ax.append(current_ax)
                if show:
                    plt.show()
                else:
                    plt.close()
        else:
            raise(ValueError('Wrong value of plot_type. Select from "performance" or "parameter"'))

        if isinstance(ax, list) and len(ax) == 1:
            ax = ax[0]
        return ax

