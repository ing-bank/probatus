import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..utils import NotFittedError
from ..utils import ApproximationWarning

import warnings


class Bucketer(object):
    def __init__(self):
        self.fitted = False

    def __repr__(self):
        repr_ = f"{self.__class__.__name__}\n\tbincount: {self.bin_count}"
        if self.fitted:
            repr_ += f"\nResults:\n\tcounts: {self.counts}\n\tboundaries: {self.boundaries}"
        return repr_

    def fit(self, x):
        """ Create bucketing on the array x

        Args:
            x: input array

        Returns: fitted bucketer object

        """
        self._fit(x)
        self.fitted = True
        return self

    def apply_bucketing(self, x_new):
        """
        Apply bucketing to new data

        Args:
            x_new: data to be bucketed

        Returns: counts of the elements in x_new using the bucketing that was obtained by fitting the Bucketer instance

        """
        if not self.fitted:
            raise NotFittedError('Bucketer is not fitted')
        else:
            # np.digitize returns the indices of the bins to which each value in input array belongs
            # the smallest value of the `boundaries` attribute equals the lowest value in the set the instance was
            # fitted on, to prevent the smallest value of x_new to be in his own bucket, we ignore the first boundary
            # value
            digitize_result = np.digitize(x_new, self.boundaries[1:], right=True)
            result = pd.DataFrame({'bucket': digitize_result}).groupby('bucket')['bucket'].count()
            # reindex the dataframe such that also empty buckets are included in the result
            result = result.reindex(np.arange(self.bin_count), fill_value=0)
            return result.values


class SimpleBucketer(Bucketer):
    """Create equally spaced bins using numpy.histogram function

    Usage:
    x = [1, 2, 1]
    bins = 3
    myBucketer = SimpleBucketer(bin_count=bins)
    myBucketer.fit(x)

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets

    """

    def __init__(self, bin_count):
        super().__init__()
        self.bin_count = bin_count

    @staticmethod
    def simple_bins(x, bin_count):
        counts, boundaries = np.histogram(x, bins=bin_count)
        return counts, boundaries

    def _fit(self, x):
        self.counts, self.boundaries = self.simple_bins(x, self.bin_count)


class AgglomerativeBucketer(Bucketer):
    """Create binning by applying the Scikit-learn implementation of Agglomerative Clustering

    Usage:
    x = [1, 2, 1]
    bins = 3
    myBucketer = AgglomerativeBucketer(bin_count=bins)
    myBucketer.fit(x)

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets

    """

    def __init__(self, bin_count):
        super().__init__()
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

    def _fit(self, x):
        self.counts, self.boundaries = self.agglomerative_clustering_binning(x, self.bin_count)


class QuantileBucketer(Bucketer):
    """Create bins with equal number of elements

    Usage:
    x = [1, 2, 1]
    bins = 3
    myBucketer = QuantileBucketer(bin_count=bins)
    myBucketer.fit(x)

    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets

    """

    def __init__(self, bin_count):
        super().__init__()
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

    def _fit(self, x):
        self.counts, self.boundaries = self.quantile_bins(x, self.bin_count)
