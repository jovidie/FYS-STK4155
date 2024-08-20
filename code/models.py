import numpy as np
from utils import mse, r2

class LinRegression:

    def __init__(self, degree) -> None:
        self._degree = degree + 1

    def _design_matrix(self, x):
        X = np.zeros((len(x), self._degree))
        X[:, 0] = 1
        for i in range(1, self._degree):
            X[:, i] = x**i
        return X

    def fit(self, x_train, y_train):
        self._X = self._design_matrix(x_train)
        self._X_T = self._X.T
        self._beta = np.linalg.inv(self._X_T @ self._X) @ self._X_T @ y_train 

    def predict(self, x_test):
        X_test = self._design_matrix(x_test)
        y_pred = X_test @ self._beta
        return y_pred 
    


if __name__ == '__main__':
    model = LinRegression(2)
    x = np.random.rand(100)
    y = 2.0 + 5*x**2 + 0.1*np.random.randn(100)

    model.fit(x, y)
    y_pred = model.predict(x)
    eps = mse(y, y_pred)
    print(eps)