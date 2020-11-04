import pytest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from probatus.interpret.shap_dependence import TreeDependencePlotter
from probatus.utils.exceptions import NotFittedError


@pytest.fixture(scope="function")
def X_y():
    return (
        pd.DataFrame(
            [
                [1.72568193, 2.21070436, 1.46039061],
                [-1.48382902, 2.88364928, 0.22323996],
                [-0.44947744, 0.85434638, -2.54486421],
                [-1.38101231, 1.77505901, -1.36000132],
                [-0.18261804, -0.25829609, 1.46925993],
                [0.27514902, 0.09608222, 0.7221381],
                [-0.27264455, 1.99366793, -2.62161046],
                [-2.81587587, 3.46459717, -0.11740999],
                [1.48374489, 0.79662903, 1.18898706],
                [-1.27251335, -1.57344342, -0.39540133],
                [0.31532891, 0.38299269, 1.29998754],
                [-2.10917352, -0.70033132, -0.89922129],
                [-2.14396343, -0.44549774, -1.80572922],
                [-3.4503348, 3.43476247, -0.74957725],
                [-1.25945582, -1.7234203, -0.77435353],
            ]
        ),
        pd.Series([1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]),
    )


@pytest.fixture(scope="function")
def expected_shap_vals():
    return pd.DataFrame(
        [
            [0.176667, 0.005833, 0.284167],
            [-0.042020, 0.224520, 0.284167],
            [-0.092020, -0.135480, -0.205833],
            [-0.092020, -0.135480, -0.205833],
            [0.002424, 0.000909, 0.263333],
            [0.176667, 0.105833, 0.284167],
            [-0.092020, -0.135480, -0.205833],
            [-0.028687, 0.311187, 0.184167],
            [0.176667, 0.005833, 0.284167],
            [-0.092020, -0.164646, -0.076667],
            [0.176667, 0.105833, 0.284167],
            [-0.092020, -0.164646, -0.176667],
            [-0.092020, -0.164646, -0.176667],
            [-0.108687, 0.081187, -0.205833],
            [-0.092020, -0.164646, -0.176667],
        ]
    )


@pytest.fixture(scope="function")
def clf(X_y):
    X, y = X_y

    model = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=5)

    model.fit(X, y)
    return model


@pytest.fixture(scope="function")
def expected_feat_importances():
    return pd.DataFrame(
        {
            "Feature Name": {0: 2, 1: 1, 2: 0},
            "Shap absolute importance": {0: 0.2199, 1: 0.1271, 2: 0.1022},
            "Shap signed importance": {0: 0.0292, 1: -0.0149, 2: -0.0076},
        }
    )


def test_not_fitted(clf):
    plotter = TreeDependencePlotter(clf)
    assert plotter.isFitted is False


def test_fit_normal(X_y, clf, expected_shap_vals):
    X, y = X_y
    plotter = TreeDependencePlotter(clf)

    plotter.fit(X, y)

    assert plotter.X.equals(X)
    assert plotter.y.equals(y)
    assert np.isclose(plotter.shap_vals_df, expected_shap_vals, atol=1e-06).all()
    assert plotter.isFitted is True


def test_get_X_y_shap_with_q_cut_normal(X_y, clf):
    X, y = X_y

    plotter = TreeDependencePlotter(clf).fit(X, y)
    plotter.min_q, plotter.max_q = 0, 1

    X_cut, y_cut, shap_val = plotter._get_X_y_shap_with_q_cut(0)
    assert np.isclose(X[0], X_cut).all()
    assert y.equals(y_cut)

    plotter.min_q = 0.2
    plotter.max_q = 0.8

    X_cut, y_cut, shap_val = plotter._get_X_y_shap_with_q_cut(0)
    assert np.isclose(
        X_cut,
        [
            -1.48382902,
            -0.44947744,
            -1.38101231,
            -0.18261804,
            0.27514902,
            -0.27264455,
            -1.27251335,
            -2.10917352,
            -1.25945582,
        ],
    ).all()
    assert np.equal(y_cut.values, [1, 0, 0, 1, 1, 0, 0, 0, 0]).all()


def test_get_X_y_shap_with_q_cut_unfitted(clf):
    plotter = TreeDependencePlotter(clf)
    with pytest.raises(NotFittedError):
        plotter._get_X_y_shap_with_q_cut(0)


def test_get_X_y_shap_with_q_cut_input(X_y, clf):
    plotter = TreeDependencePlotter(clf).fit(X_y[0], X_y[1])
    with pytest.raises(ValueError):
        plotter._get_X_y_shap_with_q_cut("not a feature")


def test_plot_normal(X_y, clf):
    plotter = TreeDependencePlotter(clf).fit(X_y[0], X_y[1])
    for binning in ["simple", "agglomerative", "quantile"]:
        fig = plotter.plot(feature=0, type_binning=binning)


def test_plot_target_names(X_y, clf):
    plotter = TreeDependencePlotter(clf).fit(X_y[0], X_y[1])
    fig = plotter.plot(feature=0, target_names=["a", "b"])
    assert plotter.target_names == ["a", "b"]


def test_plot_input(X_y, clf):
    plotter = TreeDependencePlotter(clf).fit(X_y[0], X_y[1])
    with pytest.raises(ValueError):
        plotter.plot(feature="not a feature")
    with pytest.raises(ValueError):
        plotter.plot(feature=0, type_binning=5)
    with pytest.raises(ValueError):
        plotter.plot(feature=0, min_q=1, max_q=0)


def test__repr__(clf):
    plotter = TreeDependencePlotter(clf)
    assert str(plotter) == "Shap dependence plotter for RandomForestClassifier"
