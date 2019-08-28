import pytest
import numpy as np
import pandas as pd
from pyrisk.interpret import  _shap_helpers as shap_help
from pyrisk.datasets import lending_club
from pyrisk.models import lending_club_model

from sklearn.linear_model import LogisticRegression
from pyrisk.utils import UnsupportedModelError



def get_feats_and_model():
    _, X, *dummy = lending_club()
    rf = lending_club_model()

    return rf, X


def get_shap_averages():
    # shapley values expected
    abs_avg =  [0.0024, 0.0049, 0.0048, 0.0033, 0.0008, 0.0018, 0.0015, 0.0012, 0.0016, 0.0013, 0.0027, 0.0005,
                0.0003, 0.0001, 0.0]
    avg = [0.0023, 0.0005, -0.0001, -0.0033, -0.0005, 0.0007, 0.001, 0.0001, 0.0002, -0.0, 0.0011, -0.0005, -0.0002, -0.0,
     -0.0]

    return avg, abs_avg

def test_model_support():


    rf, X = get_feats_and_model()


    shap_out = shap_help.shapley_calculator(rf,X,approximate=True)
    assert isinstance(shap_out, np.ndarray)


    #make sure that if you pass an unssupported model, the model raises the exceptions
    lr = LogisticRegression()
    with pytest.raises(UnsupportedModelError):
        assert shap_help.shapley_calculator(lr,X,approximate=True)



def test_shap_to_df():
    rf, X = get_feats_and_model()

    shap_df = shap_help.shap_to_dataframe(rf,X.head(5))

    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.columns.tolist() == X.columns.tolist()
    assert shap_df.index.tolist() == X.head(5).index.tolist()


def test_shapley_averages():
    rf, X = get_feats_and_model()

    X = X.head(5)

    shap_avg, shap_avg_abs = shap_help.compute_average_shap_per_column_raw(rf, X)

    exp_avg, exp_abs_avg = get_shap_averages()


    assert ((shap_avg.values - exp_avg) <0.0001).all()
    assert ((shap_avg_abs.values - exp_abs_avg) < 0.0001).all()

