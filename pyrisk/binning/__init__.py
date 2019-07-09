import numpy as np
import pandas as pd


def simple_bins(x, bin_count):
    """Create equaly spaced bins using numpy.histogram function

    Args:
        x (nparray) : array with a feature which has to be binned
        bin_count (int) : integer with the number of bins

    Returns:
        (list): first element of list contains the counts per bin
                second element of list contains the bin boundaries [X1,X2)

    """
    return np.histogram(x, bins=bin_count)


def quantile_bins(x, bin_count):
    """
    Create bins with equal number of elements
    :param x: array with a feature which has to be binned
    :param bin_count: integer with the number of bins
    :return: tuple containing counts per bin and the boundaries of the bins
    """
    out, bins = pd.qcut(x, q=bin_count, retbins=True)
    df = pd.DataFrame({'x': x})
    df['label'] = out
    counts = df.groupby('label').count().values.flatten()
    return counts, bins
