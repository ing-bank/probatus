"""
TODO: DOCSTRING
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
from probatus.interpret._shap_helpers import shap_to_df
from probatus.utils.arrayfuncs import assure_pandas_df
from probatus.utils.exceptions import NotFittedError


class TreeDependencePlotter:
    """
    TODO: DOCSTRING
    
    Args:
    model - classifier for which interpretation is done
    """

    def __init__(self, model):
        self.model = model

        self.isFitted = False
        self.target_names = ["target = 0", "target = 1"]

    def __repr__(self):
        return "Shap dependence for {}".format(self.model.__class__.__name__)

    def fit(self, X, y, features=None):
        """
        Fits the plotter to the model and data by computing the shap values
        
        Args:
        X (pd.DataFrame): input variables
        y (pd.Series): target variable
        features (list[str]): names of features
        
        Returns:
        self (TreeDependencePlotter)
        """
        self.X = assure_pandas_df(X)
        self.y = y

        self.features = self.X.columns if features is None else features

        self.proba = self.model.predict_proba(X)[:, 1]

        self.shap_vals_df = shap_to_df(self.model, self.X)

        self.isFitted = True
        return self

    def compute_shap_feat_importance(self, decimals=4):
        """
        TODO: DOCSTRING
        
        Returns the absolute importance and the signed importance for shapley values.

        They are ordered by decreasing absolute importance

        """
        self._check_fitted()
        if type(decimals) is not int:
            raise TypeError("decimals should be integer")
        if decimals < 0:
            raise ValueError(
                f"decimals should be greater than or equals than 0 ({decimals} was given)"
            )

        shap_abs_feat_importance = (
            self.shap_vals_df.abs().mean().sort_values(ascending=False)
        )
        shap_signed_feat_importance = self.shap_vals_df.mean()

        shap_abs_feat_importance = shap_abs_feat_importance.apply(
            lambda x: np.round(x, decimals)
        )
        shap_signed_feat_importance = shap_signed_feat_importance.apply(
            lambda x: np.round(x, decimals)
        )

        shap_abs_feat_importance.name = "Shap absolute importance"
        shap_signed_feat_importance.name = "Shap signed importance"

        out = (
            pd.concat(
                [
                    shap_abs_feat_importance,
                    shap_signed_feat_importance.iloc[shap_abs_feat_importance.index],
                ],
                axis=1,
            )
            .reset_index()
            .rename(columns={"index": "Feature Name"})
        )

        return out

    def _check_fitted(self):
        """
        Function to check if plotter is already fitted, raises exception otherwise
        """
        if not self.isFitted:
            raise NotFittedError("The plotter is not fitted yet..")

    def feature_plot(
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
        TODO: DOCSTRING
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

        fig = plt.figure(1, figsize=(10, 10))
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
        TODO: DOCSTRING
        Plot the dependence plot for one feature
        :param feature:
        :param ax:
        :param figsize:
        :return:
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

        ax.set_xlabel(feature)
        ax.set_ylabel("Shap value")
        ax.set_title(feature)
        ax.legend()

        return ax

    def _target_rate_plot(
        self, feature, bins=10, type_binning="simple", ax=None, figsize=(15, 10)
    ):
        """
        TODO: DOCSTRING
        
        Plots the distributions of the specific features, as well as the default rate as function of the feature
        :param feature:
        :param bins:
        :param type_binning: {'simple', 'agglomerative', 'quantile'}
        :param ax:
        :param figsize:
        :return:
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

        bins[-1] = bins[-1] + 1
        indices = np.digitize(x, bins)

        dfs = pd.DataFrame(
            {feature: x, "y": y, "bin_index": pd.Series(indices, index=x.index)}
        ).groupby("bin_index", as_index=True)

        def_ratio = dfs["y"].mean()
        x_vals = dfs[feature].mean()

        ax.hist(x, bins=bins, lw=2, alpha=0.4)

        ax.set_ylabel("Counts")

        ax2 = ax.twinx()

        ax2.plot(x_vals, def_ratio, color="red")
        ax2.set_ylabel("Target rate", color="red", fontsize=12)

        ax2.set_xlim(x.min(), x.max())

        return bins, ax, def_ratio

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


"""
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=15, n_features=3, n_informative=3, n_redundant=0, random_state=42
    )

    clf = RandomForestClassifier()

    clf.fit(X, y)

    bdp = TreeDependencePlotter(clf)

    bdp.fit(X, y)

    for binning in ['simple', 'agglomerative', 'quantile']:
        bdp.feature_plot(feature=2, type_binning=binning)
        plt.savefig(f"feature_plot_{binning}")

    feat_importances = bdp.compute_shap_feat_importance()

    print(feat_importances)
"""
