import numpy as np
import pandas as pd
import scipy.stats as stats

def psi(d1, d2, verbose = False, n = None, m = None):

    """
    Calculates the Population Stability Index

    A simple statistical test that quantifies the similarity of two distributions. Commonly used in the banking / risk
    modeling industry. Only works on categorical data or bucketed numerical data. Distributions must be binned/bucketed
    before passing them to this function. Distributions must have same number of buckets. Note that the PSI varies with
    number of buckets chosen (typically 10-20 bins are used).

    Args:
        d1 (np.ndarray or pd.core.series.Series) : first distribution ("expected")
        d2 (np.ndarray or pd.core.series.Series) : second distribution ("actual")
        verbose (bool)                           : print useful interpretation info to stdout (default False)
        n (int)                                  : number of samples in original d1 distribution before bucketing
        m (int)                                  : number of samples in original d2 distribution before bucketing

    Returns:
        psi_value (float) : measure of the similarity between d1 & d2. (range 0-inf, with 0 indicating identical
                            distributions and > 0.25 indicating significantly different distributions)

    """

    if isinstance(d1, pd.core.series.Series):
        d1 = np.array(d1)
    if isinstance(d2, pd.core.series.Series):
        d2 = np.array(d2)

    # Assumes same number of bins in d1 & d2
    b = len(d1)
    expected_pct = d1 / b
    actual_pct = d2 / b

    # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
    for i in range(0, b):
        if expected_pct[i] == 0:
            expected_pct[i] = 0.0001
            if verbose:
                print(f"PSI: Bucket {i} has zero counts; may result in over-estimated (larger) PSI value. Decreasing \
                        the number of buckets may also help avoid buckets with zero counts.")
    for i in range(0, b):
        if actual_pct[i] == 0:
            actual_pct[i] = 0.0001
            if verbose:
                print(f"PSI: Bucket {i} has zero counts; may result in over-estimated (larger) PSI value. Decreasing \
                        the number of buckets may also help avoid buckets with zero counts.")
        
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    if verbose:
        print('\nPSI =', psi_value)

        if psi_value <= 0.1:
            print('\nPSI <= 0.10: No significant distribution change.')
        elif 0.1 < psi_value <= 0.25:
            print('\nPSI <= 0.25: Small distribution change; may require investigation.')
        elif psi_value > 0.25:
            print('\nPSI > 0.25: Significant distribution change; investigate.')

    if verbose and n and m:
        alpha = [0.95, 0.99, 0.999]
        z_alpha = stats.norm.ppf(alpha)
        psi_critvals = ((1 / n) + (1 / m)) * (b - 1) + z_alpha * ((1 / n) + (1 / m)) * np.sqrt(2 * (b - 1))
        print('\nPSI: Critical values defined according to Yurdakul 2018')
        if psi_value > psi_critvals[2]:
            print('PSI: 99.9% confident distributions have changed.')
        elif psi_value > psi_critvals[1]:
            print('PSI: 99% confident distributions have changed.')
        elif psi_value > psi_critvals[0]:
            print('PSI: 95% confident distributions have changed.')
        elif psi_value < psi_critvals[0]:
            print('PSI: Distributions similar.')

    return psi_value
