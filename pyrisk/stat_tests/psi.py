import numpy as np
import pandas as pd


def psi(d1, d2, verbose = False):

    """Calculates the Population Stability Index, a simple statistical test that quantifies the similarity of two
    distributions. Commonly used in the banking / risk modeling industry. Only works on categorical data or bucketed
    numerical data. Distributions must be binned/bucketed before passing them to this function. Note that the PSI varies
    with number of buckets chosen.

    Args:
        d1 (np.array or pd.core.series.Series) : first distribution ("expected")
        d2 (np.array or pd.core.series.Series) : second distribution ("actual")
        verbose (bool)                         : print useful interpretation info to stdout (default False)

    Returns:
        psi_value (float) : measure of the similarity between d1 & d2. (range 0-inf, with 0 indicating identical
                            distributions and > 0.2 indicating significantly different distributions)

    """

    if isinstance(d1, pd.core.series.Series):
        d1 = np.ndarray(d1)
    if isinstance(d2, pd.core.series.Series):
        d2 = np.ndarray(d2)

    expected_pct = d1 / len(d1)
    actual_pct = d2 / len(d2)

    # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
    for i in range(0, len(expected_pct)):
        if expected_pct[i] == 0:
            expected_pct[i] = 0.0001
            if verbose:
                print(f"PSI: Bucket {i} has zero counts; may result in over-estimated (larger) PSI value. Decreasing \
                        the number of buckets may also help avoid buckets with zero counts.")
    for i in range(0, len(actual_pct)):
        if actual_pct[i] == 0:
            actual_pct[i] = 0.0001
            if verbose:
                print(f"PSI: Bucket {i} has zero counts; may result in over-estimated (larger) PSI value. Decreasing \
                        the number of buckets may also help avoid buckets with zero counts.")
        
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    if verbose:
        print('\nPSI =', psi_value)

        if psi_value < 0.1:
            print('\nPSI: No significant population change.')
        elif 0.1 <= psi_value < 0.2:
            print('\nPSI: Small population change; may require investigation.')
        elif psi_value >= 0.2:
            print('\nPSI: Significant population change.')

    return psi_value
