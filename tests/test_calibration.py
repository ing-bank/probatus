from pyrisk.calibration import *
from pyrisk.models import lending_club_model
from pyrisk.datasets import lending_club

model = lending_club_model()
credit_df, x_train, x_test, y_train, y_test = lending_club()

def test_lending_club_shape():
    assert fit_calibration(model, x_train, x_test, y_test, y_train, 2, 'sigmoid') != None
    assert fit_calibration(model, x_train, x_test, y_test, y_train, 2, 'mle') != None
    assert fit_calibration(model, x_train, x_test, y_test, y_train, 2, 'isotonic') != None

def test_mle_calibration_model():
    assert mle_calibration_model(x_train, x_test, y_test, y_train, model) != None

def test_model_calibration():
    dict_cl = model_calibration(model, x_train, x_test, y_test, y_train, ['sigmoid'], False, 10, 2)
    assert list(dict_cl.keys()) == ['sigmoid']