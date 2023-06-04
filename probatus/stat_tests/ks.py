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


from probatus.utils import NotInstalledError

try:
    from scipy import stats
except ModuleNotFoundError:
    stats = NotInstalledError("scipy", "extras")

from probatus.stat_tests.utils import verbose_p_vals

from ..utils import assure_numpy_array


@verbose_p_vals
def ks(d1, d2, verbose=False):
    """
    Calculates the Kolmogorov-Smirnov test statistic on 2 samples.

    Any binning/bucketing of the distributions/samples should be done before passing them to this function.

    References:

    - [Wikipedia article about Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
    - [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)

    Args:
        d1 (np.ndarray or pandas.Series): First sample.

        d2 (np.ndarray or pandas.Series): Second sample.

        verbose (bool): If True, useful interpretation info is printed to stdout.

    Returns:
        float: Kolmogorov-Smirnov test statistic.
        float: p-value of rejecting the null hypothesis (that the two distributions are identical).
    """
    # Perform data checks
    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    # Perform statistical tests
    ks, pvalue = stats.ks_2samp(d1, d2)

    return ks, pvalue
