from scipy import stats
from ..utils import assure_numpy_array
from probatus.stat_tests.utils import verbose_p_vals

@verbose_p_vals
def ks(d1, d2, verbose=False):
    """
    Calculates the Kolmogorov-Smirnov statistic on 2 samples. Any binning/bucketing of the distributions/samples
    should be done before passing them to this function.

    Args:
        d1 (np.ndarray or pd.core.series.Series) : first sample.

        d2 (np.ndarray or pd.core.series.Series) : second sample.

        verbose (bool)                           : helpful interpretation msgs printed to stdout (default False).

    Returns:
        (float, float): KS test stat and p-value of rejecting the null hypothesis (that the two distributions are identical)
    """

    # Perform data checks
    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    # Perform statistical tests
    ks, pvalue = stats.ks_2samp(d1, d2)

    return ks, pvalue
