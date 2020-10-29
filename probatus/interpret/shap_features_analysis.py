from probatus.interpret import TreeDependencePlotter
from sklearn.metrics import roc_auc_score
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings


class ShapFeaturesAnalyser:
    """
    This class is a wrapper that allows to easily analyse model's features. It allows to plot SHAP feature importance,
     SHAP summary plot and SHAP dependence plots.

    Example:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np

    feature_names = ['f1', 'f2', 'f3', 'f4']

    # Prepare two samples
    X, y = make_classification(n_samples=1000, n_features=4, random_state=0)
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare and fit model. Remember about class_weight="balanced" or an equivalent.
    clf = RandomForestClassifier(class_weight='balanced', n_estimators = 100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    shap_analyser = ShapFeatureAnalyser(clf, X_train, X_test, y_train, y_test)
    shap_analyser.plot('importance')
    shap_analyser.plot('summary')
    shap_analyser.plot('dependence', ['f1', 'f2'])
    ```
    """
    def __init__(self, clf, X_train, X_test, y_train, y_test, features_names=None, class_names=None, **shap_kwargs):
        """
        Initializes the class.

        Args:
            clf (binary classifier): Model fitted on X_train.
            X_train (pd.DataFrame): Dataframe containing training data.
            X_test (pd.DataFrame): Dataframe containing test data.
            y_train (pd.Series): Series of binary labels for train data.
            y_test (pd.Series): Series of binary labels for test data.
            feature_names (Optional, None, or list of str): List of feature names for the dataset. If None, then column
             names from the X_train dataframe are used.
            class_names (Optional, None, or list of str): List of class names e.g. ['neg', 'pos']. If none, the default
             ['Negative Class', 'Positive Class'] are used.
            **shap_kwargs: keyword arguments passed to shap.TreeExplainer.
        """
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Set class names
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = ['Negative Class', 'Positive Class']

        # Set feature names
        self.features_names = features_names
        if self.features_names is None:
            self.features_names = X_train.columns.tolist()

        # Calculate Metrics
        self.auc_train = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        self.auc_test = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        self.results_text = "Train AUC: {},\nTest AUC: {}.".format(
            np.round(self.auc_train, 3),
            np.round(self.auc_test, 3)
        )

        # Initialize tree dependence plotter
        self.tbp = None

        self.explainer = shap.TreeExplainer(clf, data=self.X_train, **shap_kwargs)

        # Calculate Shap values
        self.shap_values = self.explainer.shap_values(self.X_test)

        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            warnings.warn(
                'Shap values are related to the output probabilities of class 1 for this model, instead of log odds.')
            self.shap_values = self.shap_values[1]

        # Find SHAP values statistics
        shap_abs_mean = np.mean(np.abs(self.shap_values), axis=0)
        shap_mean = np.mean(self.shap_values, axis=0)

        # Prepare importance values
        self.importance_df = pd.DataFrame({
            'mean_abs_shap_value': shap_abs_mean.tolist(),
            'mean_shap_value': shap_mean.tolist()
        }, index=self.features_names).sort_values('mean_abs_shap_value', ascending=False)

    def plot(self, plot_type, target_features=None, **plot_kwargs):
        """
        Plots the appropriate SHAP plot

        Args:
            plot_type (str): One of the following:

                - 'importance': Feature importance plot, SHAP bar summary plot
                - 'summary': SHAP Summary plot
                - 'dependence': Dependence plot for each feature

            target_features (Optional, None or list of str): List of features names, for which the plots should be
             generated. If None, all features will be plotted.

            **plot_kwargs: Keyword arguments passed to the plot method. For 'importance' and 'summary' plot_type, the
             kwargs are passed to shap.summary_plot, for 'dependence' plot_type, they are passed to
             probatus.interpret.TreeDependencePlotter.feature_plot method.

        """
        if target_features is None:
            target_features = self.features_names

        target_features_indices = [self.features_names.index(target_feature) for target_feature in target_features]

        if plot_type in ['importance', 'summary']:

            # Get the target features
            target_X_test = self.X_test[target_features]
            target_shap_values = self.shap_values[:, target_features_indices]

            # Set summary plot settings
            if plot_type is 'importance':
                plot_type = 'bar'
                plot_title = 'SHAP Feature Importance'
            else:
                plot_type = 'dot'
                plot_title = 'SHAP Summary plot'

            shap.summary_plot(target_shap_values, target_X_test, plot_type=plot_type,
                              class_names=self.class_names, show=False, **plot_kwargs)
            ax = plt.gca()
            ax.set_title(plot_title)

            ax.annotate(self.results_text, (0, 0), (0, -50), fontsize=12, xycoords='axes fraction',
                        textcoords='offset points', va='top')
            plt.show()
        elif plot_type == 'dependence':
            if self.tbp is None:
                self.tdp = TreeDependencePlotter(self.clf).fit(self.X_test, self.y_test)
                ax = []
                for feature_name in target_features:
                    ax.append(
                        self.tdp.feature_plot(feature=feature_name, figsize=(10, 7), target_names=self.class_names))
                    plt.show()
                if len(ax) == 1:
                    ax = ax[0]
        else:
            raise ValueError("Wrong plot type, select from 'importance', 'summary', or 'dependence'")
        return ax