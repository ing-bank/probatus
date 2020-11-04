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
from probatus.utils.shap_helpers import shap_to_df
from probatus.utils.arrayfuncs import assure_pandas_df
from probatus.utils.exceptions import NotFittedError


class TreeDependencePlotter:
    """
    Plotter used to plot shap dependence and target rates. 
    
    Args:
        model: classifier for which interpretation is done.

    Example:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=15, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    clf = RandomForestClassifier().fit(X, y)
    bdp = TreeDependencePlotter(clf).fit(X, y)

    bdp.plot(feature=2, type_binning='simple')
    ```
    """

    def __init__(self, model):
        self.model = model

        self.isFitted = False
        self.target_names = [1, 0]

    def __repr__(self):
        return "Shap dependence plotter for {}".format(self.model.__class__.__name__)

    def fit(self, X, y, precalc_shap=None):
        """
        Fits the plotter to the model and data by computing the shap values.
        If the shap_values are passed, they do not need to be computed
        
        Args:
        X (pd.DataFrame): input variables
        y (pd.Series): target variable
        precalc_shap (Optional, None or np.array): Precalculated shap values, If provided they don't need to be
         computed.
        """
        self.X = assure_pandas_df(X)
        self.y = y
        self.features = self.X.columns

        self.shap_vals_df = shap_to_df(self.model, self.X, precalc_shap=precalc_shap)

        self.isFitted = True
        return self


    def _check_fitted(self):
        """
        Function to check if plotter is already fitted, raises exception otherwise
        
        Raises:
            NotFittedError: in case the plotter is not yet fitted.
        """
        if not self.isFitted:
            raise NotFittedError("The plotter is not fitted yet..")

    def plot(
        self,
        feature,
        figsize=(15, 10),
        bins=10,
        type_binning="simple",
        min_q=0,
        max_q=1,
        target_names=None,
    ):
        """
        Plots the shap values for data points for a given feature, as well as the target rate and values distribution.
        
        Args:
            feature (str or int): feature name of the feature to be analyzed.
            figsize ((float, float)): tuple specifying size (width, height) of resulting figure in inches.
            bins (int or list[float]): number of bins or boundaries of bins (supplied in list) for target-rate plot.
            type_binning {'simple', 'agglomerative', 'quantile'}: type of binning to be used in target-rate plot (see :mod:`binning` for more information).
            min_q (float): optional minimum quantile from which to consider values, used for plotting under outliers.
            max_q (float): optional maximum quantile until which data points are considered, used for plotting under outliers.
            target_names (list[str]): optional list of names for target classes to display in plot.
            
        Returns
            matplotlib.pyplot.Figure: Feature plot.
        """
        self._check_fitted()
        if min_q >= max_q:
            raise ValueError("min_q must be smaller than max_q")
        if feature not in self.X.columns:
            raise ValueError("Feature not recognized")
        if type_binning not in ["simple", "agglomerative", "quantile"]:
            raise ValueError(
                "Select one of the following binning methods: 'simple', 'agglomerative', 'quantile'"
            )

        if target_names is not None:
            self.target_names = target_names

        self.min_q, self.max_q = min_q, max_q

        fig = plt.figure(1, figsize=figsize)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        self._dependence_plot(feature=feature, ax=ax1)
        self._target_rate_plot(
            feature=feature, bins=bins, type_binning=type_binning, ax=ax2
        )

        ax2.set_xlim(ax1.get_xlim())

        return fig

    def _dependence_plot(self, feature, ax=None, figsize=(15, 10)):
        """
        Plots shap values for data points with respect to specified feature.
        
        Args:
            feature (str or int): feature for which dependence plot is to be created.
            ax (matplotlib.pyplot.axes): optional axis on which to draw plot.
            figsize ((float, float)): optional tuple with desired figsize in inches.
        
        Returns:
            matplotlib.pyplot.axes: axes on which plot is drawn.
        """
        if type(feature) is int:
            feature = self.features[feature]

        X, y, shap_val = self._get_X_y_shap_with_q_cut(feature=feature)

        ax.scatter(
            X[y == 0], shap_val[y == 0], label=self.target_names[0], color="lightblue"
        )

        ax.scatter(
            X[y == 1], shap_val[y == 1], label=self.target_names[1], color="darkred"
        )

        ax.set_ylabel("Shap value")
        ax.set_title(f"Dependence plot for {feature} feature")
        ax.legend()

        return ax

    def _target_rate_plot(
        self, feature, bins=10, type_binning="simple", ax=None, figsize=(15, 10)
    ):
        """ 
        Plots the distributions of the specific features, as well as the target rate as function of the feature.
        
        Args:
            feature (str or int): feature for which to create target rate plot
            bins (int or list[float]): number of bins or boundaries of desired bins in list.
            type_binning ({'simple', 'agglomerative', 'quantile'}): type of binning strategy used to create bins.
            ax (matplotlib.pyplot.axes): optional axis on which to draw plot.
            figsize ((float, float)): optional tuple with desired figsize in inches.            
            
        Returns:
            bins (list[float]): boundaries of bins used.
            ax (matplotlib.pyplot.axes): axes on which plot is drawn.
            target_ratio (float): total ratio of target (positive over negative).
        """
        x, y, shap_val = self._get_X_y_shap_with_q_cut(feature=feature)

        # Create bins if not explicitly supplied
        if type(bins) is int:
            if type_binning == "simple":
                counts, bins = SimpleBucketer.simple_bins(x, bins)
            elif type_binning == "agglomerative":
                counts, bins = AgglomerativeBucketer.agglomerative_clustering_binning(
                    x, bins
                )
            elif type_binning == "quantile":
                counts, bins = QuantileBucketer.quantile_bins(x, bins)

        # Determine bin for datapoints
        bins[-1] = bins[-1] + 1
        indices = np.digitize(x, bins)

        # Create dataframe with binned data
        dfs = pd.DataFrame(
            {feature: x, "y": y, "bin_index": pd.Series(indices, index=x.index)}
        ).groupby("bin_index", as_index=True)

        # Extract target ratio and mean feature value
        target_ratio = dfs["y"].mean()
        x_vals = dfs[feature].mean()

        # Plot target rate
        ax.hist(x, bins=bins, lw=2, alpha=0.4)
        ax.set_ylabel("Counts")
        ax2 = ax.twinx()
        ax2.plot(x_vals, target_ratio, color="red")
        ax2.set_ylabel("Target rate", color="red", fontsize=12)
        ax2.set_xlim(x.min(), x.max())
        ax.set_xlabel(f'{feature} feature values')

        return bins, ax, target_ratio

    def _get_X_y_shap_with_q_cut(self, feature):
        """
        Extracts all X, y pairs and shap values that fall within defined quantiles of the feature
        
        Args:
            feature (str): feature to return values for
        
        Returns:
            x (pd.Series): selected datapoints
            y (pd.Series): target values of selected datapoints
            shap_val (pd.Series): shap values of selected datapoints
        """
        self._check_fitted()
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
