import numpy as np
from pyrisk.binning import simple_bins


def psi(d1, d2, buckets = 10):

    """Calculates the Population Stability Index, a simple statistical test that quantifes the similarity of two 
    distributions. Commonly used in the banking / risk modeling industry. Only works on categorical data or bucketted
    numerical data.

    Args:
        d1 (np.array) : first distribution ("expected")
        d2 (np.array) : second distribution ("actual")
        buckets (int) : number of buckets (bins) for discretizing the distribution (default 10)

    Returns:
        psi_value (float) : measure of the similarity between d1 & d2. (range 0-1, with 0 indicating identical 
                            distributions and 1 indicating completely different distributions

    """

    expected_pct = simple_bins(d1, buckets)[0] / len(d1)
    actual_pct = simple_bins(d2, buckets)[0] / len(d2)

    # Necessary to avoid divide by zero. Should have negligible impact on PSI value.
    for i in range(0, len(expected_pct)):
        if expected_pct[i] == 0:
            expected_pct[i] = 0.0001
    for i in range(0, len(actual_pct)):
        if actual_pct[i] == 0:
            actual_pct[i] = 0.0001
        
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    print('PSI =', psi_value)

    if psi_value < 0.1:
        print('\nPSI: No significant population change.')
    elif 0.1 <= psi_value < 0.2:
        print('\nPSI: Moderate population change.')
    elif psi_value >= 0.2:
        print('\nPSI: Significant population change.')

    return psi_value
