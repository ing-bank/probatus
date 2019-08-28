import pandas as pd
import matplotlib.pyplot as plt
from random import sample
from scipy import stats
import seaborn as sns
from ..utils import assure_numpy_array


def shapiro_difference(d1,d2):
    """ this function examines whether the deviation of
    one random variable from normality
    is significantly different from that of the other
    random variable. If given random variables are two distinct realizations
    of the same feature, this test indicates change in distribution

    null hypothesis: distributions of given random variables are not significantly different


    inputs:
    d1:first variable
    d2:second variable

    test-statistic:
    delta: difference in shapiro values

    output:
    visualization of test outcome
    if delta is in between 0.05 and 0.095 lines in
    given distribution one cannot reject the null hypothesis
    also provides test statistic and check output

    warning: in this version the sample size of random variables should be <5000

    """

    d1 = assure_numpy_array(d1)
    d2 = assure_numpy_array(d2)

    delta = stats.shapiro(d1)[0] - stats.shapiro(d2)[0]

    MOT = pd.concat([d1, d2])
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    def ran_delta(n1, n2):
        take_ran = lambda n: random.sample(range(MOT.shape[0]), n)
        ran_1 = MOT.iloc[take_ran(n1),]
        ran_2 = MOT.iloc[take_ran(n2),]
        delta_ran = stats.shapiro(ran_1)[0] - stats.shapiro(ran_2)[0]
        return delta_ran

    collect = []
    collect = [ran_delta(n1, n2) for a in range(100)]
    collect = pd.Series(list(collect))

    quants = [0.05, 0.5, 0.95]
    sig_vals = list(collect.quantile(quants))

    fig, ax = plt.subplots(figsize=(15, 6.75))
    sns.kdeplot(collect, shade=True, color="green")
    for a in range(len(sig_vals)):
        plt.axvline(x=sig_vals[a], color="red", lw=1, ls=":")
        plt.text(sig_vals[a], 1, quants[a], fontsize=12, color="red", rotation=90)
    plt.axvline(x=delta, color="black", lw=1.5, ls='-.')
    plt.text(delta, 10, "delta", fontsize=12, color="black", rotation=90)
    plt.show()

    return (delta,sig_vals)
