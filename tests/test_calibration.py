import numpy as np
from probatus.calibration import *
from probatus.models import lending_club_model
from probatus.datasets import lending_club
from probatus.calibration import Calibrator

def test_get_calibrator():
    model = lending_club_model()
    credit_df, x_train, x_test, y_train, y_test = lending_club()
    my_calibrator = Calibrator({'sigmoid':None,'isotonic':None,'nonliear':12}, 3, 'quantile', 10)
    my_calibrator.fit(model,x_train,x_test,y_test,y_train)
    check = my_calibrator.get_calibs()
    assert list(check.keys()) == ['sigmoid', 'isotonic', 'nonliear']

def test_fit_calibrator():
    model = lending_club_model()
    credit_df, x_train, x_test, y_train, y_test = lending_club()    
    my_calibrator = Calibrator({'sigmoid':None,'isotonic':None,'nonliear':12}, 3, 'quantile', 10)
    my_calibrator.fit(model,x_train,x_test,y_test,y_train)
    check = my_calibrator.get_calibs()
    cl = check['sigmoid']
    assert np.array_equal(cl.predict_proba(x_test),my_calibrator.score('sigmoid',x_test, model))