import numpy as np
import pandas as pd
from scipy import stats


def ad(d1, d2, verbose = False):
    """
    Calculates the Anderson-Darling TS on 2 distributions.

    Can be used on continuous or discrete distributions. Any binning/bucketing of the distributions/samples should be
    done before passing them to this function.

    Anderson & Darling 1954

    Advantages:
    - Unlike the KS, the AD (like the ES) can be used on both continuous & discrete distributions.
    - Works well even when dist has fewer than 25 observations.
    - More powerful than KS, especially for differences in the tails of distributions.

    Args:
        d1 (np.array or pandas.core.series.Series) : first sample
        d2 (np.array or pandas.core.series.Series) : second sample
        verbose (bool) : helpful interpretation msgs printed to stdout (default False)

    Returns:
        ad (float)     : AD test stat
        pvalue (float) : P value of rejecting the null hypothesis (that the two distributions are identical)
    """

    if isinstance(d1, pd.core.series.Series):
        d1 = np.ndarray(d1)
    if isinstance(d2, pd.core.series.Series):
        d2 = np.ndarray(d2)

    ad, critical_values, pvalue = stats.anderson_ksamp([d1, d2])

    if verbose:
        print('\nAD: pvalue =', pvalue)

        if pvalue < 0.01:
            print('\nAD: Null hypothesis rejected with 99% confidence. Distributions very different.')
        elif pvalue < 0.05:
            print('\nAD: Null hypothesis rejected with 95% confidence. Distributions different.')
        else:
            print('\nAD: Null hypothesis cannot be rejected. Distributions not statistically different.')

    return ad, pvalue
