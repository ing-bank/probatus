import itertools
import warnings

import pandas as pd
from tqdm import tqdm

from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
from probatus.stat_tests import es, ks, psi, ad, sw


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
            'SW': Shapiro-Wilk based difference statistic
            'AD': Anderson-Darling TS


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
    myTest.compute(d1, d2, verbose=True)

    """
    statistical_test_list = ['ES', 'KS', 'PSI', 'AD', 'SW']
    binning_strategy_list = ['simplebucketer', 'agglomerativebucketer', 'quantilebucketer', None]

    def __init__(self, statistical_test, binning_strategy='simplebucketer', bin_count=10 ):
        self.statistical_test = statistical_test.upper()
        self.binning_strategy = binning_strategy
        self.bin_count = bin_count
        self.fitted = False

        if self.statistical_test.upper() not in self.statistical_test_list:
            raise NotImplementedError("The statistical test should be one of {}".format(self.statistical_test_list))
        elif self.statistical_test.upper() == 'ES':
            self.statistical_test_name = 'Epps-Singleton'
            self._statistical_test_function = es
        elif self.statistical_test.upper() == 'KS':
            self.statistical_test_name = 'Kolmogorov-Smirnov'
            self._statistical_test_function = ks
        elif self.statistical_test.upper() == 'PSI':
            self.statistical_test_name = 'Population Stability Index'
            self._statistical_test_function = psi
        elif self.statistical_test.upper() == 'SW':
            self.statistical_test_name = 'Shapiro-Wilk based difference'
            self._statistical_test_function = sw
        elif self.statistical_test.upper() == 'AD':
            self.statistical_test_name = 'Anderson-Darling TS'
            self._statistical_test_function = ad

        if self.binning_strategy:
            if self.binning_strategy.lower() not in self.binning_strategy_list:
                raise NotImplementedError("The binning strategy should be one of {}".format(self.binning_strategy_list))
            if self.binning_strategy.lower() == 'simplebucketer':
                self.binner = SimpleBucketer(bin_count=self.bin_count)
            elif self.binning_strategy.lower() == 'agglomerativebucketer':
                self.binner = AgglomerativeBucketer(bin_count=self.bin_count)
            elif self.binning_strategy.lower() == 'quantilebucketer':
                self.binner = QuantileBucketer(bin_count=self.bin_count)

    def __repr__(self):
        repr_ = "DistributionStatistics object\n\tstatistical_test: {}".format(self.statistical_test)
        if self.binning_strategy:
            repr_ += "\n\tbinning_strategy: {}\n\tbin_count: {}".format(self.binning_strategy, self.bin_count)
        else:
            repr_ += "\n\tNo binning applied"
        if self.fitted:
            repr_ += "\nResults\n\tvalue {}-statistic: {}".format(self.statistical_test, self.statistic)
        if hasattr(self, 'p_value'):
            repr_ += "\n\tp-value: {}".format(self.p_value)
        return repr_

    def compute(self, d1, d2, verbose=False, **kwargs):
        """
        Apply the statistical test and compute statistic value and p-value

        Args:
            d1: (np.array or pd.DataFrame) distribution 1
            d2: (np.array or pd.DataFrame) distribution 2
            verbose: (bool) Flag indicating whether prints should be shown

        Returns: (Touple of floats) statistic value and p_value. For PSI test the return is only statistic
        """
        if self.binning_strategy:
            self.binner.fit(d1)
            d1_preprocessed = self.binner.counts
            d2_preprocessed = self.binner.compute(d2)
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


class AutoDist(object):
    """
    Class to automatically apply all implemented statistical distribution tests and binning strategies to (a
    selection of) features in two dataframes.

    Parameters
    ----------
    statistical_tests: string 'all' or list of strings with tests to apply
        Statistical tests to apply, statistical methods implemented:
            'ES': Epps-Singleton
            'KS': Kolmogorov-Smirnov statistic
            'PSI': Population Stability Index
            'AD': Anderson-Darling TS

    binning_strategies: string 'all' or list of strings with strategies to apply
        Binning strategy to apply, binning strategies implemented:
            'SimpleBucketer': equally spaced bins
            'AgglomerativeBucketer': binning by applying the Scikit-learn implementation of Agglomerative Clustering
            'QuantileBucketer': bins with equal number of elements
            None: no binning is applied

    bin_count: integer, None or list of integers
        bin_count value(s) to be used, note that None can only be used when no bucketing strategy is applied

    Example:
        df1 = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=['feat_0', 'feat_1'])
        df2 = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=['feat_0', 'feat_1'])

        myAutoDist = AutoDist(statistical_tests='all', binning_strategies='all', bin_count=[10, 20])
        res = myAutoDist.fit(df1, df2, columns=df1.columns)
    """

    def __init__(self, statistical_tests='all', binning_strategies='all', bin_count=10):
        self.fitted = False
        if statistical_tests == 'all':
            self.statistical_tests = DistributionStatistics.statistical_test_list
        elif isinstance(statistical_tests, str):
            self.statistical_tests = [statistical_tests]
        else:
            self.statistical_tests = statistical_tests
        if binning_strategies == 'all':
            self.binning_strategies = DistributionStatistics.binning_strategy_list
        elif isinstance(binning_strategies, str):
            self.binning_strategies = [binning_strategies]
        else:
            self.binning_strategies = binning_strategies
        if not isinstance(bin_count, list):
            self.bin_count = [bin_count]
        else:
            self.bin_count = bin_count

    def __repr__(self):
        repr_ = "AutoDist object"
        if not self.fitted:
            repr_ += "\n\tAutoDist not fitted"
        if self.fitted:
            repr_ += "\n\tAutoDist fitted"
        repr_ += "\n\tstatistical_tests: {}".format(self.statistical_tests)
        repr_ += "\n\tbinning_strategies: {}".format(self.binning_strategies)
        repr_ += "\n\tbin_count: {}".format(self.bin_count)
        return repr_

    def compute(self, df1, df2, column_selection, return_failed_tests=True, suppress_warnings=True):
        """
        Fit the AutoDist object to data; i.e. apply the statistical tests and binning strategies

        Args:
            df1: dataframe 1 for distribution comparison with dataframe 2
            df2: dataframe 2 for distribution comparison with dataframe 1
            column_selection: list of columns in df1 and df2 that should be compared
            return_failed_tests: remove tests in result that did not succeed
            suppress_warnings: whether to suppress warnings during the fit process

        Returns: dataframe with results of the performed statistical tests and binning strategies

        """
        # test if all columns in column_selection are in df1 and df2
        if len(set(column_selection) - set(df1.columns)) or len(set(column_selection) - set(df2.columns)):
            raise Exception('Not all columns in `column_selection` are in the provided dataframes')

        result_all = pd.DataFrame()
        for col, stat_test, bin_strat, bins in tqdm(
                list(itertools.product(column_selection, self.statistical_tests, self.binning_strategies, self.bin_count))):
            dist = DistributionStatistics(statistical_test=stat_test, binning_strategy=bin_strat, bin_count=bins)
            try:
                if suppress_warnings:
                    warnings.filterwarnings('ignore')
                _ = dist.compute(df1[col], df2[col])
                if suppress_warnings:
                    warnings.filterwarnings('default')
                statistic = dist.statistic
                if hasattr(dist, 'p_value'):
                    p_value = dist.p_value
                else:
                    p_value = None
            except:
                statistic, p_value = 'an error occurred', None
                pass
            result_ = {'column': col, 'statistical_test': stat_test, 'binning_strategy': bin_strat, 'bin_count': bins,
                       'statistic': statistic, 'p_value': p_value}
            result_all = result_all.append(result_, ignore_index=True)
        if not return_failed_tests:
            result_all = result_all[result_all['statistic'] != 'an error occurred']
        self.fitted = True
        self._result = result_all[
            ['column', 'statistical_test', 'binning_strategy', 'bin_count', 'statistic', 'p_value']]
        self._result['bin_count'] = self._result['bin_count'].astype(int)
        self._result.loc[self._result['binning_strategy'].isnull(), 'binning_strategy'] = 'no_bucketing'

        # create pivot table as final output
        self.result = pd.pivot_table(self._result, values=['statistic', 'p_value'], index='column',
                                     columns=['statistical_test', 'binning_strategy', 'bin_count'], aggfunc='sum')
        # flatten multi-index
        self.result.columns = ["_".join([str(x) for x in line]) for line in self.result.columns.values]
        self.result.reset_index(inplace=True)
        return self.result
