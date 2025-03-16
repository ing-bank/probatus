import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from probatus.interpret.shap_dependence import DependencePlotter
from probatus.utils.exceptions import NotFittedError

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")


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
def model(X_y, random_state):
    X, y = X_y

    model = RandomForestClassifier(random_state=random_state, n_estimators=10, max_depth=5)

    model.fit(X, y)
    return model


def test_not_fitted(model, random_state):
    plotter = DependencePlotter(model, random_state)
    assert plotter.fitted is False


def test_fit_complex(complex_data_split, complex_fitted_lightgbm, random_state):
    _, X_test, _, y_test = complex_data_split

    plotter = DependencePlotter(complex_fitted_lightgbm, random_state=random_state)

    plotter.fit(X_test, y_test)

    pd.testing.assert_frame_equal(plotter.X, X_test)
    pd.testing.assert_series_equal(plotter.y, pd.Series(y_test, index=X_test.index))
    assert plotter.fitted is True

    # Check if plotting does not cause errors
    fig = plotter.plot(feature="f2_missing", show=False)
    # Verify that plot returns a Figure object
    assert isinstance(fig, matplotlib.figure.Figure)

    # Close all plots to free memory
    plt.close("all")


def test_get_X_y_shap_with_q_cut_normal(X_y, model, random_state):
    X, y = X_y

    plotter = DependencePlotter(model, random_state).fit(X, y)
    plotter.min_q, plotter.max_q = 0, 1

    X_cut, y_cut, _ = plotter._get_X_y_shap_with_q_cut(0)
    assert np.isclose(X[0], X_cut).all()
    assert y.equals(y_cut)

    plotter.min_q = 0.2
    plotter.max_q = 0.8

    X_cut, y_cut, _ = plotter._get_X_y_shap_with_q_cut(0)
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


def test_get_X_y_shap_with_q_cut_unfitted(model, random_state):
    plotter = DependencePlotter(model, random_state)
    with pytest.raises(NotFittedError):
        plotter._get_X_y_shap_with_q_cut(0)


def test_get_X_y_shap_with_q_cut_input(X_y, model, random_state):
    plotter = DependencePlotter(model, random_state).fit(X_y[0], X_y[1])
    with pytest.raises(ValueError):
        plotter._get_X_y_shap_with_q_cut("not a feature")


def test_plot_normal(X_y, model, random_state):
    plotter = DependencePlotter(model, random_state).fit(X_y[0], X_y[1])
    fig = plotter.plot(feature=0)
    # Verify that plot returns a Figure object
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_dependence_plot_returns_figure(X_y, model, random_state):
    plotter = DependencePlotter(model, random_state).fit(X_y[0], X_y[1])
    # Set required attributes that are normally set in the plot method
    plotter.min_q = 0
    plotter.max_q = 1
    plotter.alpha = 1.0
    fig = plotter._dependence_plot(feature=0)
    # Verify that _dependence_plot returns a Figure object
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_target_rate_plot_returns_figure(X_y, model, random_state):
    plotter = DependencePlotter(model, random_state).fit(X_y[0], X_y[1])
    # Set required attributes that are normally set in the plot method
    plotter.min_q = 0
    plotter.max_q = 1
    plotter.alpha = 1.0
    bin_edges, fig, target_ratio = plotter._target_rate_plot(feature=0)
    # Verify that _target_rate_plot returns a Figure object as second element of tuple
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(bin_edges, np.ndarray)
    assert isinstance(target_ratio, pd.Series)
    plt.close("all")


def test_plot_class_names(X_y, model, random_state):
    plotter = DependencePlotter(model, random_state).fit(X_y[0], X_y[1], class_names=["a", "b"])
    fig = plotter.plot(feature=0)
    assert plotter.class_names == ["a", "b"]
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_plot_input(X_y, model, random_state):
    plotter = DependencePlotter(model, random_state).fit(X_y[0], X_y[1])
    with pytest.raises(ValueError):
        plotter.plot(feature="not a feature")
    with pytest.raises(TypeError):
        plotter.plot(feature=0, bins=5.0)
    with pytest.raises(ValueError):
        plotter.plot(feature=0, min_q=1, max_q=0)


def test__repr__(model, random_state):
    """
    Test string representation.
    """
    plotter = DependencePlotter(model, random_state)
    assert str(plotter) == "Shap dependence plotter for RandomForestClassifier"
