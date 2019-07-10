from scipy import stats


def ks(d1, d2, verbose = False):
    """
    Calculates the Kolmogorov-Smirnov statistic on 2 samples. Any binning/bucketing of the distributions/samples
    should be done before passing them to this function.

    Args:
        d1 (np.array)  : first sample
        d2 (np.array)  : second sample
        verbose (bool) : helpful interpretation msgs printed to stdout (default False)

    Returns:
        ks (float)     : KS test stat
        pvalue (float) : P value of rejecting the null hypothesis (that the two distributions are identical)
    """

    ks, pvalue = stats.ks_2samp(d1, d2)

    if verbose:
        print('\nKS: pvalue =', pvalue)

        if pvalue < 0.01:
            print('\nKS: Null hypothesis rejected with 99% confidence. Distributions very different.')
        elif pvalue < 0.05:
            print('\nKS: Null hypothesis rejected with 95% confidence. Distributions different.')
        else:
            print('\nKS: Null hypothesis cannot be rejected. Distributions not statistically different.')

    return ks, pvalue
