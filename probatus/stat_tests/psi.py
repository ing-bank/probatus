import numpy as np
import scipy.stats as stats

from ..utils import assure_numpy_array


def psi(d1, d2, verbose=False):
    """
    Calculates the Population Stability Index

    A simple statistical test that quantifies the similarity of two distributions. Commonly used in the banking / risk
    modeling industry. Only works on categorical data or bucketed numerical data. Distributions must be binned/bucketed
    before passing them to this function. Bin boundaries should be same for both distributions. Distributions must have
    same number of buckets. Note that the PSI varies with number of buckets chosen (typically 10-20 bins are used).
    Quantile bucketing is typically recommended.

    Args:
        d1 (np.ndarray or pd.core.series.Series) : first distribution ("expected")
        d2 (np.ndarray or pd.core.series.Series) : second distribution ("actual")
        verbose (bool)                           : print useful interpretation info to stdout (default False)

    Returns:
        psi_value (float) : measure of the similarity between d1 & d2. (range 0-inf, with 0 indicating identical
                            distributions and > 0.25 indicating significantly different distributions)
        p_value (float): p-value

    Raises:
        UserWarning: if number of bins in d1 or d2 is less than 10 or greater than 20, where PSI is not well-behaved.
        ValueError: if d1 & d2 do not have the same number of bins
    """

    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    if len(d1) < 10:
        raise UserWarning('PSI is not well-behaved when using less than 10 bins.')
    if len(d1) > 20:
        raise UserWarning('PSI is not well-behaved when using more than 10 bins.')

    if len(d1) != len(d2):
        raise ValueError('Distributions do not have the same number of bins.')

    # Number of bins/buckets
    b = len(d1)

    n = d1.sum()
    m = d2.sum()

    expected_ratio = d1 / n
    actual_ratio = d2 / m

    # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
    for i in range(0, b):
        if expected_ratio[i] == 0:
            expected_ratio[i] = 0.0001
            if verbose:
                print(f"PSI: Bucket {i} has zero counts; may result in over-estimated (larger) PSI value. Decreasing \
                        the number of buckets may also help avoid buckets with zero counts.")
    for i in range(0, b):
        if actual_ratio[i] == 0:
            actual_ratio[i] = 0.0001
            if verbose:
                print(f"PSI: Bucket {i} has zero counts; may result in over-estimated (larger) PSI value. Decreasing \
                        the number of buckets may also help avoid buckets with zero counts.")

    psi_value = np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio))

    if verbose:
        print('\nPSI =', psi_value)

        print('\nPSI: Critical values defined according to de facto industry standard:')
        if psi_value <= 0.1:
            print('\nPSI <= 0.10: No significant distribution change.')
        elif 0.1 < psi_value <= 0.25:
            print('\nPSI <= 0.25: Small distribution change; may require investigation.')
        elif psi_value > 0.25:
            print('\nPSI > 0.25: Significant distribution change; investigate.')

        alpha = [0.95, 0.99, 0.999]
        z_alpha = stats.norm.ppf(alpha)
        psi_critvals = ((1 / n) + (1 / m)) * (b - 1) + z_alpha * ((1 / n) + (1 / m)) * np.sqrt(2 * (b - 1))
        print('\nPSI: Critical values defined according to Yurdakul 2018:')
        if psi_value > psi_critvals[2]:
            print('PSI: 99.9% confident distributions have changed.')
        elif psi_value > psi_critvals[1]:
            print('PSI: 99% confident distributions have changed.')
        elif psi_value > psi_critvals[0]:
            print('PSI: 95% confident distributions have changed.')
        elif psi_value < psi_critvals[0]:
            print('PSI: Distributions similar.')

    z = (psi_value / ((1 / n) + (1 / m)) - (b - 1)) / np.sqrt(2 * (b - 1))
    p_value = stats.norm.cdf(z)

    return psi_value, p_value
