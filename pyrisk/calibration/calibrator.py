import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from pyrisk.utils import assure_numpy_array

class Calibrator(object):

    def __init__(self, methods, folds, bins):
        """
        Initiate the calibrator

        Args:
            methods (dict) : list of calibration methods which should be applied, keys are names of the method 
                             to be applied. The values assigned to keys are the powers used for fitting the model.
                             Power applied only for MLE callibration
            folds (int) : number of folds which should be used for training the calibrator
            bins (int) : number of bins use in the calibration
            
        """           
        self.methods = methods
        self.folds = folds
        self.bins = bins
        self.cal_models = dict()

    def mle_calibration_model(self, x_train, y_train, model, power):
        """
        Train a non linear calibrated MLE model using Logistic Regression

        Args:
            model (object) : a model object which has predict_proba method
            x_train (np.array) : numpy array with features on train set
            y_train (np.array) : numpy array with target on train set
            power (int) : power which should be used in the callibrator

        Returns:
            lr_cal
            
        """   
        predIT = model.predict_proba(x_train)[:,1]
        predIT = predIT.reshape(predIT.shape[0], 1)
        lr_cal = LogisticRegression()

        features_array = predIT
        for i in range(2, power + 1):
            features_array = np.concatenate((features_array, predIT**i),axis=1)

        lr_cal.fit(features_array, y_train)

        return lr_cal

    def train_calibrator(self, model, x_train, y_train, method, power):
        """
        Train a calibration model based on selected method(s)

        Args:
            method (str) : calibration method
            model (object) : a model object which has predict_proba method
            x_train (np.array) : numpy array with features on train set
            y_train (np.array) : numpy array with target on train set
            power (int) : power which should be used in the callibrator
        Returns:
            clf
            
        """
        if method in ['sigmoid','isotonic']:
            clf = CalibratedClassifierCV(model, cv = self.folds, method = method)
            clf.fit(x_train, y_train)
        elif method == 'nonliear':
            clf = self.mle_calibration_model(x_train, y_train, model, power)

        return clf


    def calibration_plot(self, x_test, y_test, method_name, model, calibration, power):
        """
        Populate the calibration plot canvas 

        Args:
            calibration (object) : calibrated model
            x_test (np.array) : numpy array with features on test set
            y_test (np.array) : numpy array with target on test set
            bins (int) : number of bins used for calculating the calibration metrics
            method_name (str) : calibration method
            model (object) : a model object which has predict_proba method
            power (int) : power which should be used in the callibrator
            
        """

        features_array = x_test
        
        if method_name == 'nonliear':
            x_test = model.predict_proba(x_test)[:,1]
            x_test = x_test.reshape(x_test.shape[0], 1)
            features_array = x_test
            for i in range(2, power + 1):
                features_array = np.concatenate((features_array,x_test**i),axis=1)            

        y_test_predict_proba = calibration.predict_proba(features_array)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins = self.bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Calibrated - {method_name} {power}')


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
        self.model = model
        self.x_test = assure_numpy_array(x_test)
        self.y_test = assure_numpy_array(y_test)

        methods_list = list(self.methods.keys())

        for method_i in methods_list:
            power = self.methods[method_i]
            cal_clf = self.train_calibrator(model, assure_numpy_array(x_train), assure_numpy_array(y_train), method_i, power)
            self.cal_models[method_i] = cal_clf



    def plot(self):
        """
        Creting a calibration plot     
        """
        fig, ax = plt.subplots(1, figsize=(12, 6))

        y_test_predict_proba = self.model.predict_proba(self.x_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, y_test_predict_proba, n_bins = self.bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Calibrated - Original')

        methods_list = list(self.methods.keys())

        for method in methods_list:
            power = self.methods[method]
            cal_clf = self.cal_models[method]
            self.calibration_plot(self.x_test, self.y_test, method, self.model, cal_clf, power)

        
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().legend()
        plt.show()

    def score(self, method, x, model, power):
        """
        Calibrate the probabilities from the original model

        Args:
            method (string) : method used for calibration
            x (np.array) : model features which will be used for producing calibrated probabilities    
            model (object) : a model object which has predict_proba method
            power (int) : power which should be used in the callibrator
        Returns:
            cal_probas (np.array) : calibrated probabilities       
            
        """
    
        if method == 'nonliear':
            x = model.predict_proba(x)[:,1]
            x = x.reshape(x.shape[0], 1)
            for i in range(1, power + 1):
                x = np.concatenate((x,x**i),axis=1)             

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