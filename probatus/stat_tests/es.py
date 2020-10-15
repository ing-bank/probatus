# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from scipy import stats

from ..utils import assure_numpy_array
from probatus.stat_tests.utils import verbose_p_vals

@verbose_p_vals
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
        d1 (np.array or pandas.core.series.Series) : first sample.

        d2 (np.array or pandas.core.series.Series) : second sample.

        verbose (bool) : helpful interpretation msgs printed to stdout (default False).

    Returns:
        (float, float): ES test stat and p-value of rejecting the null hypothesis (that the two distributions are identical)
    """

    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    es, pvalue = stats.epps_singleton_2samp(d1, d2)

    return es, pvalue
