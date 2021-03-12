import numpy as np

from probatus.stat_tests import ks, es


def test_verbosity_true_(capsys):
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = d1
    ks(d1, d2, verbose=True)
    captured = capsys.readouterr()
    assert (
        captured.out
        == "\nKS: pvalue = 1.0\n\nKS: Null hypothesis cannot be rejected. Distributions not statistically different.\n"
    )
    es(d1, d2, verbose=True)
    captured = capsys.readouterr()
    assert (
        captured.out
        == "\nES: pvalue = 1.0\n\nES: Null hypothesis cannot be rejected. Distributions not statistically different.\n"
    )


def test_verbosity_false(capsys):
    """
    Test.
    """
    d1 = np.random.normal(size=1000)
    d2 = d1
    ks(d1, d2, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""
