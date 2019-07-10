from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

def sigmoid_calibratnio(cv = 3):

def isotonic_calibratnio():

def mle_calibration()

def calibration_plot():

def model_calibration(model, x_test, y_test, methods, calibration_plots, cv):
    """Produce callibrated probabilities based on selected method(s). Also provides model fir statistic and calibration plot 

    Args:
        model (object) : a model object which has predict_proba method
        x_test (np.array) : numpy array with features
        y_test (np.array) : numpy array with target
        methods (list) : list with method(s) which should be used for calibration
        calibration_plots (bool) : True or False if calibration plots should be produced with HS statistics
        cv (int) : numbrt of fold used in cross validation 

    Returns:
        
    """    


