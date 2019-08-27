import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
import seaborn as sns


def shadi(d1, d2):
    """ examines whether the deviation of
    one random variable from normality
    is significantly different from that of the other
    random variable. If given random variables are two distinct realizations
    of the same feature, this test indicates change in distribution

    null hypothesis: distributions of given random variables are not significantly different

    if delta is in between 0.05 and 0.095 lines in
    given distribution one cannot reject the null hypothesis
    also provides test statistic and check output

    inputs:
    d1:first variable
    d2:second variable

    test-statistic:
    delta: difference in shapiro values

    returns:
    description of the test outcome
    visualization of test outcome


    """

    d1 = pd.Series(d1)
    d2 = pd.Series(d2)

    if len(d1) > 5000:
        d1 = pd.Series(random.choices(d1, k=5000))
    if len(d2) > 5000:
        d2 = pd.Series(random.choices(d2, k=5000))

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

    collect = [ran_delta(n1, n2) for a in range(100)]
    collect = pd.Series(list(collect))

    quants = [0.025, 0.975]
    sig_vals = list(collect.quantile(quants))

    fig, ax = plt.subplots(figsize=(12, 6.75))
    sns.kdeplot(collect, shade=True, color="green")
    for a in range(len(sig_vals)):
        plt.axvline(x=sig_vals[a], color="red", lw=2, ls=":")
    plt.axvline(x=delta, color="black", lw=1.5, ls='-.')
    plt.text(delta, 20, "delta", fontsize=12, color="black", rotation=90)
    plt.show()

    print('delta value:')
    print(delta)
    print('95% confidence bounds:')
    print(sig_vals)

    if delta < sig_vals[0] or delta > sig_vals[1]:
        print('\nShapiro_Difference | Null hypothesis : <delta is not different from zero> REJECTED.')
        print('\nDelta is outside 95% CI -> Distributions very different.')
    else:
        print('\nShapiro_Difference | Null hypothesis : <delta is not different from zero> NOT REJECTED.')
        print('\nDelta is inside 95% CI -> Distributions are not different.')

    return delta, sig_vals
