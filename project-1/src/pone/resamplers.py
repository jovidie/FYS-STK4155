import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample

from pone.utils import design_matrix, mse
from pone.models import OLSRegression


class Resampler:
    """Resampler class which implements bootstrap and cross-validation."""
    def __init__(self, model, method, param):
        self._model = model
        self._method = method 
        self._param = param 
    
    def _bootstrap(self, x1, x2, y, degree):
        degrees = np.arange(degree)
        error_test = np.zeros(degree)
        bias = np.zeros(degree)
        variance = np.zeros(degree)

        X = np.column_stack((x1.ravel(), x2.ravel()))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        x1_train = X_train[:, 0]
        x2_train = X_train[:, 1]

        x1_test = X_test[:, 0]
        x2_test = X_test[:, 1]

        for p in tqdm(range(degree)):
            Xtrain = design_matrix(x1_train, x2_train, p)
            Xtest = design_matrix(x1_test, x2_test, p)

            y_pred = np.empty((y_test.shape[0], self._param))
            mse_test = np.empty(self._param)

            for i in range(self._param):
                X_, y_ = resample(Xtrain, y_train)
                self._model.fit(X_, y_)

                # y_tilde[:, i] = model.predict(X_).ravel()
                y_pred[:, i] = self._model.predict(Xtest).ravel()

                # mse_train[i] = mse(y_, y_tilde[:, i])
                mse_test[i] = mse(y_test, y_pred[:, i])

            error_test[p] = np.mean(mse_test)
            bias[p] = np.mean((y_test - np.mean(y_pred, axis=1))**2)
            variance[p] = np.mean(np.var(y_pred, axis=1, keepdims=True))
        
        return (degrees, error_test, bias, variance)
    
    def _crossval(self, x1, x2, y, degree):
        degrees = np.arange(degree)
        error_test = np.zeros(degree)

        X = np.column_stack((x1.ravel(), x2.ravel()))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        x1_train = X_train[:, 0]
        x2_train = X_train[:, 1]

        x1_test = X_test[:, 0]
        x2_test = X_test[:, 1]

        for p in tqdm(range(degree)):
            Xtrain = design_matrix(x1_train, x2_train, p)
            Xtest = design_matrix(x1_test, x2_test, p)

            self._model.fit(Xtrain, y_train)

            cross_val_score(self._model, Xtest, y_test, scoring='neg_mean_squared_error', cv=self._param)

            y_pred = self._model.predict(Xtest)
            error_test[p] = mse(y_test, y_pred)
        
        return (degrees, error_test)
    
    def run(self, x1, x2, y, degree):
        
        if self._method == 'cv':
            return self._crossval(x1, x2, y, degree)
        else:
            return self._bootstrap(x1, x2, y, degree)
        

# def bias_variance(self, x, y, z, n_bootstraps):
#     """Resampling data using bootstrap algorithm.

#     Args:
#         x (np.ndarray): x-values
#         y (np.ndarray): y-values
#         z (np.ndarray): z-values
        
#     Returns:
#         tuple: Error of train and test, bias and variance
#     """
#     X = design_matrix(x, y, self._degree)
#     X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

#     z_tilde = np.empty((z_train.shape[0], n_bootstraps))
#     z_pred = np.empty((z_test.shape[0], n_bootstraps))

#     mse_train = np.empty(n_bootstraps)
#     mse_test = np.empty(n_bootstraps)

#     for i in range(n_bootstraps):
#         X_, z_ = resample(X_train, z_train)
#         self._model.fit(X_, z_)
#         # beta = beta_ols(X_, z_)

#         z_tilde[:, i] = self._model.predict(X_).ravel() 
#         z_pred[:, i] = self._model.predict(X_test).ravel() 

#         mse_train[i] = mse(z_, z_tilde[:, i])
#         mse_test[i] = mse(z_test, z_pred[:, i])

#     error_train = np.mean(mse_train)
#     error_test = np.mean(mse_test)
#     bias = np.mean((z_test - np.mean(z_pred, axis=1))**2)
#     variance = np.mean(np.var(z_pred, axis=1, keepdims=True))

#     return (error_train, error_test, bias, variance)

# def train_loop(self, x1, x2, y):
#         p_degrees = np.arange(self._degree)
#         n_bootstraps = 100

#         # error_train = np.empty(degree)
#         error_test = np.zeros(self._degree)
#         bias = np.zeros(self._degree)
#         variance = np.zeros(self._degree)

#         X = np.column_stack((x1, x2))
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#         x1_train = X_train[:, 0]
#         x2_train = X_train[:, 1]

#         x1_test = X_test[:, 0]
#         x2_test = X_test[:, 1]

#         for p in range(self._degree):
#             Xtrain = design_matrix([x1_train, x2_train], p)
#             Xtest = design_matrix([x1_test, x2_test], p)

#             # y_tilde = np.empty((y_train.shape[0], n_bootstraps))
#             y_pred = np.empty((y_test.shape[0], n_bootstraps))

#             # mse_train = np.empty(n_bootstraps)
#             mse_test = np.empty(n_bootstraps)

#             for i in range(n_bootstraps):
#                 X_, y_ = resample(Xtrain, y_train)
#                 # model = LinearRegression()
#                 self._model.fit(X_, y_)

#                 # y_tilde[:, i] = model.predict(X_).ravel()
#                 y_pred[:, i] = self._model.predict(Xtest).ravel()

#                 # mse_train[i] = mse(y_, y_tilde[:, i])
#                 mse_test[i] = mse(y_test, y_pred[:, i])

#             error_test[p] = np.mean(mse_test)
#             bias[p] = np.mean((y_test - np.mean(y_pred, axis=1))**2)
#             variance[p] = np.mean(np.var(y_pred, axis=1, keepdims=True))

#         return (p_degrees, error_test, bias, variance)