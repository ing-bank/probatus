from pyrisk.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
from pyrisk.stat_tests import es, ks, psi


class DistributionStatistics(object):
    """
    Wrapper that applies a statistical method and a binning strategy to data.

    Parameters
    ----------
    statistical_test: string
        Statistical method to apply, statistical methods implemented:
            'ES': Epps-Singleton
            'KS': Kolmogorov-Smirnov statistic
            'PSI': Population Stability Index

    binning_strategy: string or None
        Binning strategy to apply, binning strategies implemented:
            'SimpleBucketer': equally spaced bins
            'AgglomerativeBucketer': binning by applying the Scikit-learn implementation of Agglomerative Clustering
            'QuantileBucketer': bins with equal number of elements

    bin_count: integer or None
        In case binning_strategy is not None, specify the number of bins to be used by the binning strategy

    Example:
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.normal(size=1000), 10)[0]

    myTest = DistributionStatistics('KS', 'SimpleBucketer', bin_count=10)
    myTest.fit(d1, d2, verbose=True)

    """

    def __init__(self, statistical_test, binning_strategy, bin_count=None):
        self.statistical_test = statistical_test.upper()
        self.binning_strategy = binning_strategy
        self.bin_count = bin_count
        self.fitted = False

        if self.statistical_test.upper() not in ['ES', 'KS', 'PSI']:
            raise NotImplementedError("The statistical test should be one of 'ES', 'KS', 'PSI'")
        elif self.statistical_test.upper() == 'ES':
            self._statistical_test_function = es
        elif self.statistical_test.upper() == 'KS':
            self._statistical_test_function = ks
        elif self.statistical_test.upper() == 'PSI':
            self._statistical_test_function = psi

        if self.binning_strategy:
            if self.binning_strategy.lower() not in ['simplebucketer', 'agglomerativebucketer', 'quantilebucketer',
                                                     None]:
                raise NotImplementedError(
                    "The binning strategy should be one of 'SimpleBucketer', 'AgglomerativeBucketer', "
                    "'QuantileBucketer' "
                    "or None")
            if self.binning_strategy.lower() == 'simplebucketer':
                self.binner = SimpleBucketer(bin_count=self.bin_count)
            elif self.binning_strategy.lower() == 'agglomerativebucketer':
                self.binner = AgglomerativeBucketer(bin_count=self.bin_count)
            elif self.binning_strategy.lower() == 'quantilebucketer':
                self.binner = QuantileBucketer(bin_count=self.bin_count)

    def __repr__(self):
        repr_ = f"DistributionStatistics object\n\tstatistical_test: {self.statistical_test}"
        if self.binning_strategy:
            repr_ += f"\n\tbinning_strategy: {self.binning_strategy}\n\tbin_count: {self.bin_count}"
        else:
            repr_ += "\n\tNo binning applied"
        if self.fitted:
            repr_ += f"\nResults\n\tvalue {self.statistical_test}-statistic: {self.statistic}"
        if hasattr(self, 'p_value'):
            repr_ += f"\n\tp-value: {self.p_value}"
        return repr_

    def fit(self, d1, d2, verbose=False, **kwargs):
        """
        Fit the DistributionStatistics object to data; i.e. apply the statistical test

        Args:
            d1: distribution 1
            d2: distribution 2
            verbose:
            **kwargs:
                for PSI specific: set `n` and `m` as the size of the dataset *before* bucketing

        Returns: statistic value and p_value (if available, e.g. not for PSI)

        """
        if self.binning_strategy:
            self.binner.fit(d1)
            d1_preprocessed = self.binner.counts
            self.binner.fit(d2)
            d2_preprocessed = self.binner.counts
        else:
            d1_preprocessed, d2_preprocessed = d1, d2

        res = self._statistical_test_function(d1_preprocessed, d2_preprocessed, verbose=verbose, **kwargs)
        self.fitted = True
        if type(res) == tuple:
            self.statistic, self.p_value = res
            return self.statistic, self.p_value
        else:
            self.statistic = res
            return self.statistic
