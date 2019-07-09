import numpy as np

def simple_bins(x, bin_count):
    """Create equaly spaced bins using numpy.histogram function

    Args:
        x (nparray) : array with a feature which has to be binned
        bin_count (int) : integer with the number of bins

    Returns:
        (list): first element of list contains the counts per bin 
                second element of list contains the bin boundaries [X1,X2)

    """
    return np.histogram(x,bins = bin_count)