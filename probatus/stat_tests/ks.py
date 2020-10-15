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
