import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def simple_bins(x, bin_count):
    """Create equaly spaced bins using numpy.histogram function

    Args:
        x (nparray) : array with a feature which has to be binned
        bin_count (int) : integer with the number of bins

    Returns:
        (list): first element of list contains the counts per bin
                second element of list contains the bin boundaries [X1,X2)

    """
    counts, boundaries = np.histogram(x, bins=bin_count)
    return counts, boundaries


def agglomerative_clustering_binning(x, bin_count):
    """
    Create binning by applying the Scikit-learn implementation of Agglomerative Clustering
    :param x: array with a feature which has to be binned
    :param bin_count: integer with the number of bins
    :return: tuple consisting of list of counts per bin and list of bin boundaries
    """
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
