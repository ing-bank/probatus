from pyrisk.binning import simple_bins
from scipy import stats


def ks(d1, d2, buckets = 0, verbose = False):
    """Calculates the Kolmogorov-Smirnov statistic on 2 samples.

    Args:
        d1 (np.array)  : first sample
        d2 (np.array)  : second sample
        buckets (int)  : number of buckets (bins) for discretizing the distribution (default 0)
        verbose (bool) : helpful interpretation msgs printed to stdout (default False)

    Returns:
        pvalue (float) : P value of rejecting the null hypothesis (that the two distributions are identical)

    """

    if buckets == 0:
        ks, pvalue = stats.ks_2samp(d1, d2)
    else:
        d1_bucketed = simple_bins(d1, buckets)[0]
        d2_bucketed = simple_bins(d2, buckets)[0]
        ks, pvalue = stats.ks_2samp(d1_bucketed, d2_bucketed)

    if verbose:
        print('\nKS: pvalue =', pvalue)

        if pvalue < 0.01:
            print('\nKS: Null hypothesis rejected with 99% confidence. Distributions very different.')
        elif pvalue < 0.05:
            print('\nKS: Null hypothesis rejected with 95% confidence. Distributions different.')
        else:
            print('\nKS: Null hypothesis cannot be rejected. Distributions not statistically different.')

    return pvalue
