import numpy as np
from sklearn.linear_model import Lasso


class OLSRegression:
    """Ordinary Least Square regression model."""
    def __init__(self) -> None:
        self._beta = None
    
    @property
    def beta(self):
        return self._beta

    def fit(self, X_train, y_train):
        XT = X_train.T
        XTX = XT @ X_train
        self._beta = np.linalg.pinv(XTX) @ XT @ y_train 

    def predict(self, X_test):
        y_pred = X_test @ self._beta
        return y_pred  


class RidgeRegression:
    """Regression model with Ridge regularization"""
    def __init__(self, lmbda) -> None:
        self._lmbda = lmbda
        self._beta = None

    @property
    def beta(self):
        return self._beta

    def fit(self, X_train, y_train):
        XT = X_train.T
        XTX = XT @ X_train
        self._beta = np.linalg.pinv(XTX + self._lmbda*np.eye(len(XTX))) @ XT @ y_train 
        return self

    def predict(self, X_test):
        y_pred = X_test @ self._beta
        return y_pred 


class LassoRegression:
    """Regression model with Lasso regularization, wraps sklearn model."""
    def __init__(self, lmbda, iter=10_000) -> None:
        self._model = Lasso(fit_intercept=True, max_iter=iter, alpha=lmbda)
        self._beta = None

    @property
    def beta(self):
        return self._beta
        
    def fit(self, X_train, y_train):
        self._beta = self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)
 