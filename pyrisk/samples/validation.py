import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

from .. utils import assure_numpy_array,UnsupportedModelError




class ModelBased(object):

    def __init__(self, model_type='rf', keep_samples=False, **model_kwargs):
        """

        Args:
            model_type:
            keep_samples:
            **model_kwargs:
        """

        self.fitted = False
        self.model = None
        self._keep_samples = keep_samples
        self.col_names = None

        if model_type =='rf':
            self.model = RandomForestClassifier(**model_kwargs)
        elif model_type =='lr':
            self.model = LogisticRegression(**model_kwargs)
        else:
            self.model = model_type


    def __repr__(self):
        repr_ = f"{self.__class__.__name__}\n\tModel type: {self.get_model_name()}"
        if self.fitted:
            repr_ += "\nThe model predicts the samples with an AUC of {:.3f}".format(self.auc_test)
        return repr_

    def get_model_name(self):
        return self.model.__class__.__name__


    def _get_feature_importance(self):
        """Returns feature importance for a given model

        Returns:
            importances (pandas series): feature importance
        """

        try:
            importances = self.model.feature_importances_
        except:
            try:
                importances = self.model.coef_
            except:
                raise UnsupportedModelError("Model type {} is not supported".format(self.get_model_name()))


        if self.col_names is not None:
            feat_importances = pd.Series(importances, index=self.col_names)
        else:
            feat_importances = pd.Series(importances)

        return feat_importances

    def _check_and_fill_missings(self, X, impute_method=True):
        """
        Checks if for missing values: if there are and impute_method=True
        it fills them with zero

        Args:
            X (numpy array):
            impute_method (bool)
        Returns:
            X (numpy array)
        """
        if np.isnan(X).any() == True:
            warnings.warn("You have missing values in your sample",
                          DeprecationWarning,
                          stacklevel=2)
        if impute_method:
            warnings.warn("Going to temporary solution: filling missing values with zero",
                          DeprecationWarning,
                          stacklevel=2)
            X[np.isnan(X)] = 0
        return X


    def _propensity_check(self, X1, X2, test_size=0.33, random_state=None):
        """Calculates and returns the AUC score.

        Args:
            X1 (numpy.ndarray)          : first sample
            X2 (numpy.ndarray)          : second sample

        Returns:
            AUC score                   : float
        """
        col_names=None
        if type(X1)==pd.DataFrame:
            self.col_names = X1.columns

        X1 = assure_numpy_array(X1)
        X2 = assure_numpy_array(X2)

        X1 = self._check_and_fill_missings(X1)
        X2 = self._check_and_fill_missings(X2)

        X = np.concatenate([X1,X2])
        y = np.zeros(len(X1) + len(X2))
        y[len(X1):] = 1
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

        if self._keep_samples:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train =  y_train
            self.y_test = y_test

        self.model.fit(X_train,y_train)


        auc_train = roc_auc_score(y_train, self.model.predict_proba(X_train)[:,1])
        auc_test = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])


        importances = self._get_feature_importance()

        return auc_train, auc_test, importances

    def fit(self, X1,X2):

        self.auc_train, self.auc_test, self.importances = self._propensity_check(X1,X2)
        self.fitted=True

        return self





