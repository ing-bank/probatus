import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

from .. utils import assure_numpy_array, UnsupportedModelError, class_name_from_object




class ResemblanceModel(object):

    def __init__(self, model_type='rf', keep_samples=False, **model_kwargs):
        """
        This model checks for similarity of two samples.

        It assignes to each sample a label, reshuffles the values and randomly selects a fraction of the sample mix for
        training the model, and the remainder to estimate how well it can distinguish the samples.

        If the samples are coming from the same distribution, you would expect an AUC on the test to be of the order
        of 0.5, i.e. it is not possible to distingusih the samples

        Args:
            model_type: (str or model) define which model you want to run for resemblance modeling.
            Currently supported:
                - 'rf' (default): RandomForestClassifier from sklearn
                - 'lr': LogisticRegression from sklearn

                You can define your custom model object:
                Example:
                    from xgboost import XGBoostClassifier

                    ResemblanceModel(model_type=XGBoostClassifier()).fit(X1,X2)

            keep_samples: boolean, default is False. Set it to True if you want to retrieve the samples used by
                the underlying model to fit. Those samples are build from merging the X1 and X2 samples and then randomly
                sampling.
                If true, the samples are available as properties:
                    - X_train: features used for training the underlying model
                    - y_train: label of the samples used
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
            self.model = model_type(**model_kwargs)


    def __repr__(self):
        repr_ = f"{self.__class__.__name__}\n\tUnderlying model type: {self.get_model_name()}"
        if self.fitted:
            repr_ += "\nThe model is able to distinguish the samples with an AUC of {:.3f}".format(self.auc_test)
        return repr_

    def get_model_name(self):
        return class_name_from_object(self.model)


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
                importances = importances.reshape(importances.shape[1],)

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
            X1 (numpy.ndarray): first sample
            X2 (numpy.ndarray): second sample
            test_size: size of the test sample (fraction)


        Returns:
            auc score on the training sample, auc on the test sample, and feature imporatances  : float
        """

        assert X1.shape[1] == X2.shape[1], "The number of columns in X1 and X2 does not match!"
        assert type(X1) == type(X2), "Non matching types for  X1 and X2"

        if type(X1)==pd.DataFrame:
            self.col_names = X1.columns
            assert X1.columns.tolist() == X2.columns.tolist(), "The columns of the DataFrame X1 and X2 do not match"

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

    def fit(self, X1,X2, **kwargs):
        """
            Fit the underlying model to compare the samples X1 or X2


        Args:
            X1 (numpy.ndarray or pd.DataFrame): first sample
            X2 (numpy.ndarray or pd.DataFrame): second sample
            **kwargs:
                test_size (float, default=0.33): fraction to use for test samples
                random_state=None: random state of train test split

        """


        self.auc_train, self.auc_test, self.importances = self._propensity_check(X1,X2,**kwargs)
        self.fitted=True

        return self





