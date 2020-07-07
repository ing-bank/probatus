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
            [[ 1.72568193,  2.21070436,  1.46039061],
             [-1.48382902,  2.88364928,  0.22323996],
             [-0.44947744,  0.85434638, -2.54486421],
             [-1.38101231,  1.77505901, -1.36000132],
             [-0.18261804, -0.25829609,  1.46925993],
             [ 0.27514902,  0.09608222,  0.7221381 ],
             [-0.27264455,  1.99366793, -2.62161046],
             [-2.81587587,  3.46459717, -0.11740999],
             [ 1.48374489,  0.79662903,  1.18898706],
             [-1.27251335, -1.57344342, -0.39540133],
             [ 0.31532891,  0.38299269,  1.29998754],
             [-2.10917352, -0.70033132, -0.89922129],
             [-2.14396343, -0.44549774, -1.80572922],
             [-3.4503348 ,  3.43476247, -0.74957725],
             [-1.25945582, -1.7234203,  -0.77435353]]
        ),
        pd.Series([1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]),
    )

@pytest.fixture(scope="function")
def expected_shap_vals():
    return pd.DataFrame(
        [[ 0.19862036,  0.0344507,   0.24426228],
         [-0.06764518,  0.13746226,  0.27751626],
         [-0.14957961, -0.0441542,  -0.23893285],
         [-0.14797009, -0.03603992, -0.23865666],
         [ 0.08948749, -0.00153468,  0.24938052],
         [ 0.19829059,  0.07319157,  0.24585117],
         [-0.12474354, -0.03249663, -0.2354265 ],
         [-0.05132364,  0.14994058,  0.21871639],
         [ 0.2097446,   0.0500709,   0.24751783],
         [-0.13677833, -0.13156015, -0.08432819],
         [ 0.19729059,  0.07419157,  0.24585117],
         [-0.13436034, -0.1060588,  -0.21224753],
         [-0.13780478, -0.07541991, -0.21944197],
         [-0.14328407,  0.04549881, -0.24488141],
         [-0.14224634, -0.09829079, -0.20212953]]
    )


@pytest.fixture(scope="function")
def clf(X_y):
    X, y = X_y
    
    model = RandomForestClassifier(random_state=42)
    
    model.fit(X, y)
    return model


def test_not_fitted(clf):
    plotter = TreeDependencePlotter(clf)
    assert(plotter.isFitted is False)

def test_fit_normal(X_y, clf, expected_shap_vals):
    X, y = X_y
    plotter = TreeDependencePlotter(clf)
    
    plotter.fit(X, y)

    assert(plotter.X.equals(X))
    assert(plotter.y.equals(y))
    assert(np.isclose(plotter.proba, [0.94, 0.81, 0.03, 0.04, 0.8 , 0.98, 0.07, 0.78, 0.97, 0.11, 0.98, 0.01, 0.03, 0.12, 0.02]).all())
    assert(np.isclose(plotter.shap_vals_df, expected_shap_vals).all())
    assert(plotter.isFitted is True)  
       
def test_fit_features(X_y, clf):
    X, y = X_y
    plotter = TreeDependencePlotter(clf)
    
    feature_names = ['feature un', 'feature dos', 'feature tres']
    plotter.fit(X, y, feature_names)
    
    assert plotter.features == feature_names
    
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
    assert np.isclose(X_cut, [-1.48382902, -0.44947744, -1.38101231, -0.18261804,  0.27514902, -0.27264455, -1.27251335, -2.10917352, -1.25945582]).all()
    assert np.equal(y_cut.values, [1, 0, 0, 1, 1, 0, 0, 0, 0]).all()
    
def test_get_X_y_shap_with_q_cut_unfitted(clf):
    plotter = TreeDependencePlotter(clf)
    with pytest.raises(NotFittedError):
        plotter._get_X_y_shap_with_q_cut(0)
