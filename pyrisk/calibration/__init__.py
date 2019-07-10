import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def fit_calibration(model, x_train, x_test, y_test, y_train, folds, method):
    """
    Produce calibrated probabilities based on selected method(s)

    Args:
        model (object) : a model object which has predict_proba method
        x_test (np.array) : numpy array with features on test set
        y_test (np.array) : numpy array with target on test set
        x_train (np.array) : numpy array with features on train set
        y_train (np.array) : numpy array with target on train set
        folds (int) : numbrt of fold used in cross validation 
        method (str) : calibration method

    Returns:
        clf
        
    """
    if method in ['sigmoid','isotonic']:
        clf = CalibratedClassifierCV(model, cv = folds, method=method)
        clf.fit(x_train, y_train.values)
    elif method == 'mle':
        clf = mle_calibration_model(x_train, x_test, y_test, y_train, model)

    return clf

def mle_calibration_model(x_train, x_test, y_test, y_train, model):
    """
    Train a calibrated MLE model using Logistic Regression

    Args:
        model (object) : a model object which has predict_proba method
        x_test (np.array) : numpy array with features on test set
        y_test (np.array) : numpy array with target on test set
        x_train (np.array) : numpy array with features on train set
        y_train (np.array) : numpy array with target on train set

    Returns:
        lr_cal
        
    """   
    predIT = model.predict_proba(x_train)[:,1]
    predIT = predIT.reshape(predIT.shape[0], 1)
    lr_cal = LogisticRegression()    
    lr_cal.fit(np.concatenate((predIT,predIT**2,predIT**3),axis=1), y_train.values)

    return lr_cal



def calibration_plot(calibration, x_test, y_test, bins, method, model):
    """
    Produce a calibration plot 

    Args:
        calibration (object) : calibrated model
        x_test (np.array) : numpy array with features on test set
        y_test (np.array) : numpy array with target on test set
        bins (int) : number of bins used for calculating the calibration metrics
        method (str) : calibration method
        model (object) : a model object which has predict_proba method

    Returns:
        cal_models (dict) : 
         
    """
    if method == 'mle':
        x_test = model.predict_proba(x_test)[:,1]
        x_test = x_test.reshape(x_test.shape[0], 1)
        x_test = np.concatenate((x_test,x_test**2,x_test**3),axis=1)

    y_test_predict_proba = calibration.predict_proba(x_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins = bins)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Calibrated - {method}')


def model_calibration(model, x_train, x_test, y_test, y_train, methods, calibration_plots, folds, bins):
    """
    Produce calibrated probabilities based on selected method(s). Also provides model fit statistic and calibration plot 

    Args:
        model (object) : a model object which has predict_proba method
        x_test (np.array) : numpy array with features on test set
        y_test (np.array) : numpy array with target on test set
        x_train (np.array) : numpy array with features on train set
        y_train (np.array) : numpy array with target on train set
        methods (list) : list with method(s) which should be used for calibration
        calibration_plots (bool) : True or False if calibration plots should be produced with HS statistics
        folds (int) : numbrt of fold used in cross validation 
        bins (int) : number of bins used for calculating the calibration metrics

    Returns:
        cal_models (dict) : dictionary which holds calibrated clasifiers
         
    """
    if calibration_plots:
        fig, ax = plt.subplots(1, figsize=(12, 6))

    cal_models = dict()

    for method_i in methods:
        cal_clf = fit_calibration(model, x_train, x_test, y_test, y_train, folds, method_i)
        cal_models[method_i] = cal_clf

        if calibration_plots:
            calibration_plot(cal_clf,x_test, y_test, bins, method_i, model)
    
    if calibration_plots:
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().legend()
        plt.show()

    return cal_models









