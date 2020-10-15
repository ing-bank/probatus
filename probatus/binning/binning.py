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
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from probatus.utils import assure_numpy_array, TreePathFinder, ApproximationWarning

import warnings
from abc import ABC, abstractmethod

class Bucketer(ABC):

    def __repr__(self):
        repr_ = f"{self.__class__.__name__}\n\tbincount: {self.bin_count}"
        if hasattr(self, "boundaries_"):
            repr_ += f"\nResults:\n\tcounts: {self.counts_}\n\tboundaries: {self.boundaries_}"
        return repr_

    @abstractmethod
    def fit(self):
        pass

    @property
    def boundaries(self):
        warnings.warn("The 'boundaries' attribute is deprecated, use 'boundaries_' instead. The underscore suffix signals this is a fitted attribute.",
                       DeprecationWarning)
        check_is_fitted(self)
        return self.boundaries_

    @property
    def counts(self):
        warnings.warn("The 'counts' attribute is deprecated, use 'counts_' instead. The underscore suffix signals this is a fitted attribute.", 
            DeprecationWarning)
        check_is_fitted(self)
        return self.counts_

    def compute(self, X, y=None):
        """
        Applies fitted bucketing algorithm on input data and counts number of samples per bin

        Args:
            X: (np.array) data to be bucketed
            y: (np.array) ignored, for sklearn compatibility

        Returns: counts of the elements in X using the bucketing that was obtained by fitting the Bucketer instance

        """
        check_is_fitted(self)
        
        # np.digitize returns the indices of the bins to which each value in input array belongs
        # the smallest value of the `boundaries` attribute equals the lowest value in the set the instance was
        # fitted on, to prevent the smallest value of x_new to be in his own bucket, we ignore the first boundary
        # value
        digitize_result = np.digitize(X, self.boundaries_[1:], right=True)
        result = pd.DataFrame({'bucket': digitize_result}).groupby('bucket')['bucket'].count()
        # reindex the dataframe such that also empty buckets are included in the result
        result = result.reindex(np.arange(self.bin_count), fill_value=0)
        return result.values

    def fit_compute(self, X, y = None):
        """
        Apply bucketing to new data and return number of samples per bin

        Args:
            X: (np.array) data to be bucketed
            y: (np.array) One dimensional array, used if the target is needed for the bucketing. By default is set to
            None

        Returns: counts of the elements in x_new using the bucketing that was obtained by fitting the Bucketer instance

        """
        self.fit(X, y)
        return self.compute(X, y)


class SimpleBucketer(Bucketer):
    """Create equally spaced bins using numpy.histogram function

    Usage:
    ```python
    x = [1, 2, 1]
    bins = 3
    myBucketer = SimpleBucketer(bin_count=bins)
    myBucketer.fit(x)
    ```

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets
    """
    def __init__(self, bin_count):
        self.bin_count = bin_count

    @staticmethod
    def simple_bins(x, bin_count):
        counts, boundaries = np.histogram(x, bins=bin_count)
        return counts, boundaries

    def fit(self, x, y=None):
        """
        Fit bucketing on x

        Args:
            x: (np.array) Input array on which the boundaries of bins are fitted
            y: (np.array) ignored. For sklearn-compatibility

        Returns: fitted bucketer object
        """
        self.counts_, self.boundaries_ = self.simple_bins(x, self.bin_count)
        return self


class AgglomerativeBucketer(Bucketer):
    """Create binning by applying the Scikit-learn implementation of Agglomerative Clustering

    Usage:
    ```python
    x = [1, 2, 1]
    bins = 3
    myBucketer = AgglomerativeBucketer(bin_count=bins)
    myBucketer.fit(x)
    ```

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets
    """
    def __init__(self, bin_count):
        self.bin_count = bin_count

    @staticmethod
    def agglomerative_clustering_binning(x, bin_count):
        clustering = AgglomerativeClustering(n_clusters=bin_count).fit(np.asarray(x).reshape(-1, 1))
        df = pd.DataFrame({'x': x, 'label': clustering.labels_}).sort_values(by='x')
        cluster_minimum_values = df.groupby('label')['x'].min().sort_values().tolist()
        cluster_maximum_values = df.groupby('label')['x'].max().sort_values().tolist()
        # take the mean of the upper boundary of a cluster and the lower boundary of the next cluster
        boundaries = [np.mean([cluster_minimum_values[i + 1], cluster_maximum_values[i]]) for i in
                      range(len(cluster_minimum_values) - 1)]
        # add the lower boundary of the lowest cluster and the upper boundary of the highest cluster
        boundaries = [cluster_minimum_values[0]] + boundaries + [cluster_maximum_values[-1]]
        counts = df.groupby('label')['label'].count().values
        return counts, boundaries

    def fit(self, x, y=None):
        """
        Fit bucketing on x

        Args:
            x: (np.array) Input array on which the boundaries of bins are fitted
            y: (np.array) ignored. For sklearn-compatibility

        Returns: fitted bucketer object
        """
        self.counts_, self.boundaries_ = self.agglomerative_clustering_binning(x, self.bin_count)
        return self


class QuantileBucketer(Bucketer):
    """Create bins with equal number of elements

    Usage:
    ```python
    x = [1, 2, 1]
    bins = 3
    myBucketer = QuantileBucketer(bin_count=bins)
    myBucketer.fit(x)
    ```

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets
    """
    def __init__(self, bin_count):
        self.bin_count = bin_count

    @staticmethod
    def quantile_bins(x, bin_count, inf_edges=False):

        try:
            out, boundaries = pd.qcut(x, q=bin_count, retbins=True, duplicates='raise')
        except ValueError:
            # If there are too many duplicate values (assume a lot of filled missings)
            # this crashes - the exception drops them.
            # This means that it will return approximate quantile bins
            out, boundaries = pd.qcut(x, q=bin_count, retbins=True, duplicates='drop')
            warnings.warn(ApproximationWarning("Approximated quantiles - too many unique values" ))
        df = pd.DataFrame({'x': x})
        df['label'] = out
        counts = df.groupby('label').count().values.flatten()
        if inf_edges:
            boundaries[0] = -np.inf
            boundaries[-1] = np.inf
        return counts, boundaries

    def fit(self, x, y=None):
        """
        Fit bucketing on x

        Args:
            x: (np.array) Input array on which the boundaries of bins are fitted
            y: (np.array) ignored. For sklearn-compatibility

        Returns: fitted bucketer object
        """
        self.counts_, self.boundaries_ = self.quantile_bins(x, self.bin_count)
        return self


class TreeBucketer(Bucketer):
    """Class for bucketing using Decision Trees.
    It returns the optimal buckets found by a one-dimensional Decision Tree relative to a binary target.

    Useful if the buckets be defined such that there is a substantial difference between the buckets in
    the distribution of the target.
    
    Usage:
    ```python
    x = [1, 2, 2, 5 ,3]
    y = [0, 0 ,1 ,1 ,1]
    bins = 3
    myBucketer = TreeBucketer(inf_edges=True,max_depth=2,min_impurity_decrease=0.001)
    myBucketer.fit(x,y)
    ```

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets

    Args:
        inf_edges (boolean): Flag to keep the lower and upper boundary as infinite (if set to True).
        If false, the edges will be set to the minimum and maximum value of the fitted

        tree (sklearn.tree.DecisionTreeClassifier): decision tree object defined by the user. By default is None, and
        it will be constructed using tkhe provided **kwargs

        **tree_kwargs: kwargs related to the decision tree.
            For and extensive list of parameteres, please check the sklearn Decision Tree Classifier documentation
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

            The most relevant parameteres useful for the bucketing, are listed below:


                - criterion : {"gini", "entropy"}, default="gini"
                    The function to measure the quality of a split. Supported criteria are
                    "gini" for the Gini impurity and "entropy" for the information gain.


                - max_depth : int, default=None
                    Defines the maximum theoretical number of bins (2^max_depth)

                    The maximum depth of the tree. If None, then nodes are expanded until
                    all leaves are pure or until all leaves contain less than
                    min_samples_split samples.



                - min_samples_leaf : int or float, default=1
                    Defines the minimum number of entries in each bucket.

                    The minimum number of samples required to be at a leaf node.
                    A split point at any depth will only be considered if it leaves at
                    least ``min_samples_leaf`` training samples in each of the left and
                    right branches.  This may have the effect of smoothing the model,
                    especially in regression.

                    - If int, then consider `min_samples_leaf` as the minimum number.
                    - If float, then `min_samples_leaf` is a fraction and
                      `ceil(min_samples_leaf * n_samples)` are the minimum
                      number of samples for each node.

                    .. versionchanged:: 0.18
                       Added float values for fractions.



                min_impurity_decrease : float, default=0.0
                    Controls the way the TreeBucketer splits.
                    When the criterion is set to 'entropy', the best results tend to
                    be achieved in the range [0.0001 - 0.01]

                    A node will be split if this split induces a decrease of the impurity
                    greater than or equal to this value.

                    The weighted impurity decrease equation is the following::

                        N_t / N * (impurity - N_t_R / N_t * right_impurity
                                            - N_t_L / N_t * left_impurity)

                    where ``N`` is the total number of samples, ``N_t`` is the number of
                    samples at the current node, ``N_t_L`` is the number of samples in the
                    left child, and ``N_t_R`` is the number of samples in the right child.

                    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
                    if ``sample_weight`` is passed.

                    .. versionadded:: 0.19

    """

    def __init__(self, inf_edges = False, tree = None, **tree_kwargs):
        self.bin_count = -1
        self.inf_edges = inf_edges
        if tree is None:
            self.tree = DecisionTreeClassifier(**tree_kwargs)
        else:
            self.tree = tree


    @staticmethod
    def tree_bins(x, y, inf_edges, tree):

        X_in = assure_numpy_array(x).reshape(-1, 1)
        y_in = assure_numpy_array(y).reshape(-1, 1)
        tree.fit(X_in,y_in)

        if tree.min_samples_leaf>=X_in.shape[0]:
            error_msg = (
                "Cannot Fit decision tree. min_samples_leaf must be < than the length of x.m" +
                f"Currently min_samples_leaf {tree.min_samples_leaf} " +
                f"and the length of X is {X_in.shape[0]}"
            )
            raise ValueError(error_msg)

        leaves = tree.apply(X_in)
        index, counts = np.unique(leaves, return_counts=True)

        bin_count = len(index)

        tpf = TreePathFinder(tree)
        boundaries = [bound['min'] for bound in tpf.get_boundaries().values()]
        boundaries += [tpf.get_boundaries()[leaves[-1]]['max']]

        if not inf_edges:
            boundaries[0] = np.min(X_in)
            boundaries[-1] = np.max(X_in)

        return counts, boundaries, bin_count, tree

    def fit(self, X, y):
        """
        Fit bucketing on x

        Args:
            x: (np.array) Input array on which the boundaries of bins are fitted
            y: (np.array) optional, One dimensional array with the target.

        Returns: fitted bucketer object
        """
        self.counts_, self.boundaries_, self.bin_count, self.tree = self.tree_bins(X,y, self.inf_edges, self.tree)
        return self