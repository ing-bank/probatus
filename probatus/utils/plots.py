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


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def plot_distributions_of_feature(
    feature_distributions,
    feature_name=None,
    sample_names=None,
    plot_bw_method=0.05,
    plot_perc_outliers_removed=0.01,
    plot_figsize=(15, 6),
):
    """
    This function plots multiple distributions of the same feature.

    It is e.g. useful to compare
    distribution between train and test.

    For categorical feature the plot bar is plotted, and for numeric the density plot.

    Args:
        feature_distributions (list of pd.Series): List of distributions of the same feature,
        e.g. values of feature 'f1' for Train, Validation and Test.

        feature_name (Optional, str): Name of the feature plotted.

        sample_names (Optional, list of str): List of names of samples e.g. ['Train', 'Validation', 'Test'].

        plot_bw_method (Optional, float): Estimator bandwidth in density plot.

        plot_perc_outliers_removed (Optional, float): Percentage of outliers removed from each side before plotting.

        plot_figsize (Optional, tuple): Size of the figure.

    """
    figure(figsize=plot_figsize)

    if feature_name is None:
        feature_name = feature_distributions[0].name

    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(len(feature_distributions))]

    if feature_distributions[0].dtype.name == "category":
        data_dict = {}

        for feature_distribution_index in range(len(feature_distributions)):
            data_dict[sample_names[feature_distribution_index]] = feature_distributions[
                feature_distribution_index
            ].value_counts(normalize=True)

        plt.ylabel("Relative frequencies of values in feature.")
    else:
        for feature_distribution_index in range(len(feature_distributions)):
            current_feature = feature_distributions[feature_distribution_index]

            # Remove outliers in each feature
            current_feature = current_feature[
                current_feature.between(
                    current_feature.quantile(0 + plot_perc_outliers_removed),
                    current_feature.quantile(1 - plot_perc_outliers_removed),
                )
            ]

            # Plot density plot
            current_feature.plot.density(bw_method=plot_bw_method)

    plt.title(f"Distribution of {feature_name}")
    plt.xlabel("Feature Distribution")
    plt.legend(sample_names, loc="upper right")
    plt.show()
