from pyrisk.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
from pyrisk.stat_tests import es, ks, psi


class DistributionStatistics(object):
    """
    Wrapper that applies a statistical method and a binning strategy to data.

    Statistical methods implemented:
        'es': Epps-Singleton
        'ks': Kolmogorov-Smirnov statistic
        'psi': Population Stability Index

    Binning strategies implemented:
        'SimpleBucketer': equally spaced bins
        'AgglomerativeBucketer': binning by applying the Scikit-learn implementation of Agglomerative Clustering
        'QuantileBucketer': bins with equal number of elements

    Usage:
    d1 = np.histogram(np.random.normal(size=1000), 10)[0]
    d2 = np.histogram(np.random.normal(size=1000), 10)[0]

    myTest = DistributionStatistics('ks', 'SimpleBucketer', bin_count=10)
    myTest.fit(d1, d2, verbose=True)

    """

    def __init__(self, statistical_test, binning_strategy, bin_count):
        self.statistical_test = statistical_test
        self.binning_strategy = binning_strategy
        self.bin_count = bin_count
        self.fitted = False

        if self.statistical_test.lower() not in ['es', 'ks', 'psi']:
            raise NotImplementedError("The statistical test should be one of 'es', 'ks', 'psi'")
        elif self.statistical_test.lower() == 'es':
            self._statistical_test_function = es
        elif self.statistical_test.lower() == 'ks':
            self._statistical_test_function = ks
        elif self.statistical_test.lower() == 'psi':
            self._statistical_test_function = psi

        if self.binning_strategy.lower() not in ['simplebucketer', 'agglomerativebucketer', 'quantilebucketer']:
            raise NotImplementedError(
                "The binning strategy should be one of 'SimpleBucketer', 'AgglomerativeBucketer', 'QuantileBucketer'")
        if self.binning_strategy.lower() == 'simplebucketer':
            self.myBinner = SimpleBucketer(bin_count=self.bin_count)
        elif self.binning_strategy.lower() == 'agglomerativebucketer':
            self.myBinner = AgglomerativeBucketer(bin_count=self.bin_count)
        elif self.binning_strategy.lower() == 'quantilebucketer':
            self.myBinner = QuantileBucketer(bin_count=self.bin_count)

    def __repr__(self):
        repr_ = f"DistributionStatistics object\n\tstatistical_test: {self.statistical_test}\n\tbinning_strategy: {self.binning_strategy}\n\tbin_count: {self.bin_count}"
        if self.fitted:
            repr_ += f"\nResults\n\tvalue {self.statistical_test} statistic: {self.statistic}"
            if hasattr(self, 'p_value'):
                repr_ += f"\n\tp value: {self.p_value}"
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
        if self._statistical_test_function == psi:
            if not 'n' in kwargs or not 'm' in kwargs:
                raise IOError('For PSI please specify the length of unbinned d1 and d2')
        self.myBinner.fit(d1)
        d1_binned = self.myBinner.counts
        self.myBinner.fit(d2)
        d2_binned = self.myBinner.counts

        res = self._statistical_test_function(d1_binned, d2_binned, verbose=verbose, **kwargs)
        self.fitted = True
        if type(res) == tuple:
            self.statistic, self.p_value = res
            return self.statistic, self.p_value
        else:
            self.statistic = res
            return self.statistic
