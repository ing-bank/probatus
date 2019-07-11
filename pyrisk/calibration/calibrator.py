import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class Calibrator(object):

    def __init__(self, methods, folds, bins):
        """
        Initiate the calibrator

        Args:
            methods (list) : list of calibration methods which should be applied
            folds (int) : number of folds which should be used for training the calibrator
            bins (int) : number of bins use in the calibration
            
        """           
        self.methods = methods
        self.folds = folds
        self.bins = bins
        self.cal_models = dict()

    def mle_calibration_model(self, x_train, y_train, model):
        """
        Train a non linear calibrated MLE model using Logistic Regression

        Args:
            model (object) : a model object which has predict_proba method
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

    def train_calibrator(self, model, x_train, y_train, method):
        """
        Train a calibration model based on selected method(s)

        Args:
            method (str) : calibration method
            model (object) : a model object which has predict_proba method
            x_train (np.array) : numpy array with features on train set
            y_train (np.array) : numpy array with target on train set
        Returns:
            clf
            
        """
        if method in ['sigmoid','isotonic']:
            clf = CalibratedClassifierCV(model, cv = self.folds, method = method)
            clf.fit(x_train, y_train.values)
        elif method == 'nonliear':
            clf = self.mle_calibration_model(x_train, y_train, model)

        return clf


    def calibration_plot(self, x_test, y_test, method, model, calibration):
        """
        Populate the calibration plot canvas 

        Args:
            calibration (object) : calibrated model
            x_test (np.array) : numpy array with features on test set
            y_test (np.array) : numpy array with target on test set
            bins (int) : number of bins used for calculating the calibration metrics
            method (str) : calibration method
            model (object) : a model object which has predict_proba method
            
        """
        if method == 'nonliear':
            x_test = model.predict_proba(x_test)[:,1]
            x_test = x_test.reshape(x_test.shape[0], 1)
            x_test = np.concatenate((x_test,x_test**2,x_test**3),axis=1)

        y_test_predict_proba = calibration.predict_proba(x_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins = self.bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Calibrated - {method}')


    def fit(self, model, x_train, x_test, y_test, y_train):
        """
        Train calibrators which can be used for calibrating probabilities. Storing all the trained calibrators in the dictionary

        Args:
            model (object) : a model object which has predict_proba method
            x_train (np.array) : numpy array with features on train set
            y_train (np.array) : numpy array with target on train set
            x_test (np.array) : numpy array with features on test set
            y_test (np.array) : numpy array with target on test set            

        """

        for method_i in self.methods:
            cal_clf = self.train_calibrator(model, x_train, y_train, method_i)
            self.cal_models[method_i] = cal_clf

        self.model = model
        self.x_test = x_test
        self.y_test = y_test


    def plot(self):
        """
        Creting a calibration plot     
        """
        fig, ax = plt.subplots(1, figsize=(12, 6))

        y_test_predict_proba = self.model.predict_proba(self.x_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, y_test_predict_proba, n_bins = self.bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Calibrated - Original')

        for method in self.methods:
            cal_clf = self.cal_models[method]
            self.calibration_plot(self.x_test, self.y_test, method, self.model, cal_clf)

        
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().legend()
        plt.show()

    def score(self, method, x, model):
        """
        Calibrate the probabilities from the original model

        Args:
            method (string) : method used for calibration
            x (np.array) : model features which will be used for producing calibrated probabilities    
            model (object) : a model object which has predict_proba method
        Returns:
            cal_probas (np.array) : calibrated probabilities       
            
        """
    
        if method == 'nonliear':
            x = model.predict_proba(x)[:,1]
            x = x.reshape(x.shape[0], 1)
            x = np.concatenate((x,x**2,x**3),axis=1)

        cal_clf = self.cal_models[method] 
        cal_probas = cal_clf.predict_proba(x)

        return cal_probas

    def get_calibs(self):
        """
        Returns calibrated models
        Returns:
            cal_models (dics) : dictionary with calibrated models           
        """

        return self.cal_models