from scipy import stats
from ..utils import assure_numpy_array


def es(d1, d2, verbose=False):
    """
    Calculates the Epps-Singleton test statistic on 2 distributions. Can be used on continuous or discrete
    distributions. Any binning/bucketing of the distributions/samples should be done before passing them to this
    function.

    Whereas KS relies on the empirical distribution function, ES is based on the empirical characteristic function
    (Epps & Singleton 1986, Goerg & Kaiser 2009).

    Advantages:
    - Unlike the KS, the ES can be used on both continuous & discrete distributions.
    - ES has higher power (vs KS) in many examples.
    Disadvantages:
    - Not recommended for fewer than 25 observations. Instead, use the Anderson-Darling TS. (However, ES can still be
      used for small samples. A correction factor is applied so that the asymptotic TS distribution more closely follows
      the chi-squared distribution, such that p-values can be computed.)

    Args:
        d1 (np.array or pandas.core.series.Series) : first sample
        d2 (np.array or pandas.core.series.Series) : second sample
        verbose (bool) : helpful interpretation msgs printed to stdout (default False)

    Returns:
        es (float)     : ES test stat
        pvalue (float) : P value of rejecting the null hypothesis (that the two distributions are identical)
    """

    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    es, pvalue = stats.epps_singleton_2samp(d1, d2)

    if verbose:
        print('\nES: pvalue =', pvalue)

        if pvalue < 0.01:
            print('\nES: Null hypothesis rejected with 99% confidence. Distributions very different.')
        elif pvalue < 0.05:
            print('\nES: Null hypothesis rejected with 95% confidence. Distributions different.')
        else:
            print('\nES: Null hypothesis cannot be rejected. Distributions not statistically different.')

    return es, pvalue
