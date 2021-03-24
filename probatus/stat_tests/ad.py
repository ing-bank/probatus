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


from ..utils import assure_numpy_array
from probatus.stat_tests.utils import verbose_p_vals
from probatus.utils import NotInstalledError

try:
    from scipy import stats
except ModuleNotFoundError:
    stats = NotInstalledError("scipy", "extras")


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
        (float, float): AD test stat and p-value of rejecting the null hypothesis
                       (that the two distributions are identical)
    """
    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    ad, critical_values, pvalue = stats.anderson_ksamp([d1, d2])

    return ad, pvalue
