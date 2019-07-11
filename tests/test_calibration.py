import numpy as np
from pyrisk.calibration import *
from pyrisk.models import lending_club_model
from pyrisk.datasets import lending_club
from pyrisk.calibration import Calibrator

def test_get_calibrator():
    model = lending_club_model()
    credit_df, x_train, x_test, y_train, y_test = lending_club()
    my_calibrator = Calibrator(methods = ['sigmoid','isotonic','nonliear'], folds = 3, bins =10)
    my_calibrator.fit(model,x_train,x_test,y_test,y_train)
    check = my_calibrator.get_calibs()
    assert check['sigmoid'] != None

def test_fit_calibrator():
    model = lending_club_model()
    credit_df, x_train, x_test, y_train, y_test = lending_club()    
    my_calibrator = Calibrator(methods = ['sigmoid','isotonic','nonliear'], folds = 3, bins =10)
    my_calibrator.fit(model,x_train,x_test,y_test,y_train)
    check = my_calibrator.get_calibs()
    cl = check['sigmoid']
    assert np.array_equal(cl.predict_proba(x_test),my_calibrator.score('sigmoid',x_test, model))