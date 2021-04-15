# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
from probatus.utils import (
    BaseFitComputePlotClass,
    shap_to_df,
    preprocess_data,
    preprocess_labels,
)


class DependencePlotter(BaseFitComputePlotClass):
    """
    Plotter used to plot SHAP dependence plot together with the target rates.

    Currently it supports tree-based and linear models.

    Args:
        model: classifier for which interpretation is done.

    Example:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from probatus.interpret import DependencePlotter

    X, y = make_classification(n_samples=15, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    clf = RandomForestClassifier().fit(X, y)
    bdp = DependencePlotter(clf)
    shap_values = bdp.fit_compute(X, y)

    bdp.plot(feature=2, type_binning='simple')
    ```

    <img src="../img/model_interpret_dep.png"/>
    """

    def __init__(self, clf, verbose=0):
        """
        Initializes the class.

        Args:
            clf (model object):
                Binary classification model or pipeline.

            verbose (int, optional):
                Controls verbosity of the output:

                - 0 - neither prints nor warnings are shown
                - 1 - 50 - only most important warnings regarding data properties are shown (excluding SHAP warnings)
                - 51 - 100 - shows most important warnings, prints of the feature removal process
                - above 100 - presents all prints and all warnings (including SHAP warnings).
        """
        self.clf = clf
        self.verbose = verbose

    def __repr__(self):
        """
        Represent string method.
        """
        return "Shap dependence plotter for {}".format(self.clf.__class__.__name__)

    def fit(self, X, y, column_names=None, class_names=None, precalc_shap=None, **shap_kwargs):
        """
        Fits the plotter to the model and data by computing the shap values.

        If the shap_values are passed, they do not need to be computed.

        Args:
            X (pd.DataFrame): input variables.

            y (pd.Series): target variable.

            column_names (None, or list of str, optional):
                List of feature names for the dataset. If None, then column names from the X_train dataframe are used.

            class_names (None, or list of str, optional):
                List of class names e.g. ['neg', 'pos']. If none, the default ['Negative Class', 'Positive Class'] are
                used.

            precalc_shap (Optional, None or np.array):
                Precalculated shap values, If provided they don't need to be computed.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.
        """
        self.X, self.column_names = preprocess_data(X, X_name="X", column_names=column_names, verbose=self.verbose)
        self.y = preprocess_labels(y, y_name="y", index=self.X.index, verbose=self.verbose)

        # Set class names
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = ["Negative Class", "Positive Class"]

        self.shap_vals_df = shap_to_df(self.clf, self.X, precalc_shap=precalc_shap, verbose=self.verbose, **shap_kwargs)

        self.fitted = True
        return self

    def compute(self):
        """
        Computes the report returned to the user, namely the SHAP values generated on the dataset.

        Returns:
            (pd.DataFrame):
                SHAP Values for X.
        """
        self._check_if_fitted()
        return self.shap_vals_df

    def fit_compute(self, X, y, column_names=None, class_names=None, precalc_shap=None, **shap_kwargs):
        """
        Fits the plotter to the model and data by computing the shap values.

        If the shap_values are passed, they do not need to be computed

        Args:
            X (pd.DataFrame):
                Provided dataset.

            y (pd.Series):
                Binary labels for X.

            column_names (None, or list of str, optional):
                List of feature names for the dataset. If None, then column names from the X_train dataframe are used.

            class_names (None, or list of str, optional):
                List of class names e.g. ['neg', 'pos']. If none, the default ['Negative Class', 'Positive Class'] are
                used.

            precalc_shap (Optional, None or np.array):
                Precalculated shap values, If provided they don't need to be computed.

            **shap_kwargs:
                keyword arguments passed to
                [shap.Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer).
                It also enables `approximate` and `check_additivity` parameters, passed while calculating SHAP values.
                The `approximate=True` causes less accurate, but faster SHAP values calculation, while
                `check_additivity=False` disables the additivity check inside SHAP.

        Returns:
            (pd.DataFrame):
                SHAP Values for X.
        """
        self.fit(X, y, column_names=column_names, class_names=class_names, precalc_shap=precalc_shap, **shap_kwargs)
        return self.compute()

    def plot(
        self,
        feature,
        figsize=(15, 10),
        bins=10,
        type_binning="simple",
        show=True,
        min_q=0,
        max_q=1,
    ):
        """
        Plots the shap values for data points for a given feature, as well as the target rate and values distribution.

        Args:
            feature (str or int):
                Feature name of the feature to be analyzed.

            figsize ((float, float), optional):
                Tuple specifying size (width, height) of resulting figure in inches.

            bins (int or list[float]):
                Number of bins or boundaries of bins (supplied in list) for target-rate plot.

            type_binning ({'simple', 'agglomerative', 'quantile'}):
                Type of binning to be used in target-rate plot (see :mod:`binning` for more information).

            show (bool, optional):
                If True, the plots are showed to the user, otherwise they are not shown. Not showing plot can be useful,
                when you want to edit the returned axis, before showing it.

            min_q (float, optional):
                Optional minimum quantile from which to consider values, used for plotting under outliers.

            max_q (float, optional):
                Optional maximum quantile until which data points are considered, used for plotting under outliers.

        Returns
            (list(matplotlib.axes)):
                List of axes that include the plots.
        """
        self._check_if_fitted()
        if min_q >= max_q:
            raise ValueError("min_q must be smaller than max_q")
        if feature not in self.X.columns:
            raise ValueError("Feature not recognized")
        if type_binning not in ["simple", "agglomerative", "quantile"]:
            raise ValueError("Select one of the following binning methods: 'simple', 'agglomerative', 'quantile'")

        self.min_q, self.max_q = min_q, max_q

        _ = plt.figure(1, figsize=figsize)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        self._dependence_plot(feature=feature, ax=ax1)
        self._target_rate_plot(feature=feature, bins=bins, type_binning=type_binning, ax=ax2)

        ax2.set_xlim(ax1.get_xlim())

        if show:
            plt.show()
        else:
            plt.close()

        return [ax1, ax2]

    def _dependence_plot(self, feature, ax=None):
        """
        Plots shap values for data points with respect to specified feature.

        Args:
            feature (str or int):
                Feature for which dependence plot is to be created.

            ax (matplotlib.pyplot.axes, optional):
                Optional axis on which to draw plot.

        Returns:
            (matplotlib.pyplot.axes):
                Axes on which plot is drawn.
        """
        if type(feature) is int:
            feature = self.column_names[feature]

        X, y, shap_val = self._get_X_y_shap_with_q_cut(feature=feature)

        ax.scatter(X[y == 0], shap_val[y == 0], label=self.class_names[0], color="lightblue")

        ax.scatter(X[y == 1], shap_val[y == 1], label=self.class_names[1], color="darkred")

        ax.set_ylabel("Shap value")
        ax.set_title(f"Dependence plot for {feature} feature")
        ax.legend()

        return ax

    def _target_rate_plot(self, feature, bins=10, type_binning="simple", ax=None):
        """
        Plots the distributions of the specific features, as well as the target rate as function of the feature.

        Args:
            feature (str or int):
                Feature for which to create target rate plot.

            bins (int or list[float]), optional:
                Number of bins or boundaries of desired bins in list.

            type_binning ({'simple', 'agglomerative', 'quantile'}, optional):
                Type of binning strategy used to create bins.

            ax (matplotlib.pyplot.axes, optional):
                Optional axis on which to draw plot.

        Returns:
            (list[float], matplotlib.pyplot.axes, float):
                Tuple of boundaries of bins used, axis on which plot is drawn, total ratio of target (positive over
                negative).
        """
        x, y, shap_val = self._get_X_y_shap_with_q_cut(feature=feature)

        # Create bins if not explicitly supplied
        if type(bins) is int:
            if type_binning == "simple":
                counts, bins = SimpleBucketer.simple_bins(x, bins)
            elif type_binning == "agglomerative":
                counts, bins = AgglomerativeBucketer.agglomerative_clustering_binning(x, bins)
            elif type_binning == "quantile":
                counts, bins = QuantileBucketer.quantile_bins(x, bins)

        # Determine bin for datapoints
        bins[-1] = bins[-1] + 1
        indices = np.digitize(x, bins)
        # Create dataframe with binned data
        dfs = pd.DataFrame({feature: x, "y": y, "bin_index": pd.Series(indices, index=x.index)}).groupby(
            "bin_index", as_index=True
        )

        # Extract target ratio and mean feature value
        target_ratio = dfs["y"].mean()
        x_vals = dfs[feature].mean()

        # Transform the first and last bin to work with plt.hist method
        if bins[0] == -np.inf:
            bins[0] = x.min()
        if bins[-1] == np.inf:
            bins[-1] = x.max()

        # Plot target rate
        ax.hist(x, bins=bins, lw=2, alpha=0.4)
        ax.set_ylabel("Counts")
        ax2 = ax.twinx()
        ax2.plot(x_vals, target_ratio, color="red")
        ax2.set_ylabel("Target rate", color="red", fontsize=12)
        ax2.set_xlim(x.min(), x.max())
        ax.set_xlabel(f"{feature} feature values")

        return bins, ax, target_ratio

    def _get_X_y_shap_with_q_cut(self, feature):
        """
        Extracts all X, y pairs and shap values that fall within defined quantiles of the feature.

        Args:
            feature (str): feature to return values for

        Returns:
            x (pd.Series): selected datapoints
            y (pd.Series): target values of selected datapoints
            shap_val (pd.Series): shap values of selected datapoints
        """
        self._check_if_fitted()
        if feature not in self.X.columns:
            raise ValueError("Feature not found in data")

        # Prepare arrays
        x = self.X[feature]
        y = self.y
        shap_val = self.shap_vals_df[feature]

        # Determine quantile ranges
        x_min = x.quantile(self.min_q)
        x_max = x.quantile(self.max_q)

        # Create filter
        filter = (x >= x_min) & (x <= x_max)

        # Filter and return terms
        return x[filter], y[filter], shap_val[filter]
