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


import warnings

import numpy as np

from probatus.utils import NotInstalledError

try:
    from scipy import stats
except ModuleNotFoundError:
    stats = NotInstalledError("scipy", "extras")

from ..utils import assure_numpy_array


def psi(d1, d2, verbose=False):
    """
    Calculates the Population Stability Index.

    A simple statistical test that quantifies the similarity of two distributions. Commonly used in the banking / risk
    modeling industry. Only works on categorical data or bucketed numerical data. Distributions must be binned/bucketed
    before passing them to this function. Bin boundaries should be same for both distributions. Distributions must have
    same number of buckets. Note that the PSI varies with number of buckets chosen (typically 10-20 bins are used).
    Quantile bucketing is typically recommended.

    Args:
        d1 (np.ndarray or pd.core.series.Series) : first distribution ("expected").

        d2 (np.ndarray or pd.core.series.Series) : second distribution ("actual").

        verbose (bool)                           : print useful interpretation info to stdout (default False).

    Returns:
        (float, float) : measure of the similarity between d1 & d2. (range 0-inf, with 0 indicating identical
        distributions and > 0.25 indicating significantly different distributions); p_value for rejecting null
        hypothesis (that the two distributions are identical).
    """
    # Perform data checks
    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    if len(d1) < 10:
        warnings.warn("PSI is not well-behaved when using less than 10 bins.")
    if len(d1) > 20:
        warnings.warn("PSI is not well-behaved when using more than 20 bins.")
    if len(d1) != len(d2):
        raise ValueError("Distributions do not have the same number of bins.")

    # Number of bins/buckets
    b = len(d1)

    # Calculate the number of samples in each distribution
    n = d1.sum()
    m = d2.sum()

    # Calculate the ratio of samples in each bin
    expected_ratio = d1 / n
    actual_ratio = d2 / m

    # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
    has_empty_bucket = False
    for i in range(b):
        if expected_ratio[i] == 0:
            expected_ratio[i] = 0.0001
            has_empty_bucket = True

        if actual_ratio[i] == 0:
            actual_ratio[i] = 0.0001
            has_empty_bucket = True

    if has_empty_bucket:
        warnings.warn(
            "PSI: Some of the buckets have zero counts. In theory this situation would mean PSI=Inf due to "
            "division by 0. However, we artificially modified the count of samples in these bins to a small "
            "number. This may cause that the PSI value for this feature is over-estimated (larger). "
            "Decreasing the number of buckets may also help avoid buckets with zero counts."
        )

    # Calculate the PSI value
    psi_value = np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio))

    # Print the evaluation of statistical hypotheses
    if verbose:
        print("\nPSI =", psi_value)

        print("\nPSI: Critical values defined according to de facto industry standard:")
        if psi_value <= 0.1:
            print("PSI <= 0.10: No significant distribution change.")
        elif 0.1 < psi_value <= 0.25:
            print("PSI <= 0.25: Small distribution change; may require investigation.")
        elif psi_value > 0.25:
            print("PSI > 0.25: Significant distribution change; investigate.")

        # Calculate the critical values and
        alpha = [0.95, 0.99, 0.999]
        z_alpha = stats.norm.ppf(alpha)
        psi_critvals = ((1 / n) + (1 / m)) * (b - 1) + z_alpha * ((1 / n) + (1 / m)) * np.sqrt(2 * (b - 1))
        print("\nPSI: Critical values defined according to Yurdakul (2018):")
        if psi_value > psi_critvals[2]:
            print("99.9% confident distributions have changed.")
        elif psi_value > psi_critvals[1]:
            print("99% confident distributions have changed.")
        elif psi_value > psi_critvals[0]:
            print("95% confident distributions have changed.")
        elif psi_value < psi_critvals[0]:
            print("No significant distribution change.")

    # Calculate p-value
    z = (psi_value / ((1 / n) + (1 / m)) - (b - 1)) / np.sqrt(2 * (b - 1))
    p_value = stats.norm.cdf(z)

    return psi_value, p_value
