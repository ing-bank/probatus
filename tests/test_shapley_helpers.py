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


