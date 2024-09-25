from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np

from pone.utils import design_matrix, franke_function

class Regression(ABC):
    """Abstract class for regression models."""
    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass


class LinearRegression(Regression):

    def __init__(self) -> None:
        self._beta = None
    
    @property
    def beta(self):
        return self._beta

    def fit(self, X_train, y_train):
        XT = X_train.T
        XTX = XT @ X_train
        self._beta = np.linalg.pinv(XTX) @ XT @ y_train 
        return self

    def predict(self, X_test):
        y_pred = X_test @ self._beta
        return y_pred 


class RidgeRegression(Regression):

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


class LassoRegression(Regression):

    def __init__(self, lmbda) -> None:
        self._model = Lasso(lmbda)
        self._beta = None

    @property
    def beta(self):
        return self._beta
        
    def fit(self, X_train, y_train):
        # Needs implementation for Lasso
        # XT = X_train.T
        # XTX = XT @ X_train
        # self._beta = np.linalg.pinv(XTX + self._lmbda*np.eye(len(XTX))) @ XT @ y_train 
        self._beta = self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self._model.predict(X_test)