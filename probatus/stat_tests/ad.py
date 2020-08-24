from scipy import stats
from ..utils import assure_numpy_array
from probatus.stat_tests.utils import verbose_p_vals

@verbose_p_vals
def ad(d1, d2, verbose=False):
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
        d1 (np.array or pandas.core.series.Series): first sample

        d2 (np.array or pandas.core.series.Series): second sample

        verbose (bool): helpful interpretation msgs printed to stdout (default False)

    Returns:
        (float, float): AD test stat and p-value of rejecting the null hypothesis (that the two distributions are identical)
    """

    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    ad, critical_values, pvalue = stats.anderson_ksamp([d1, d2])

    return ad, pvalue
