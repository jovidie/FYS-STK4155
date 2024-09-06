import numpy as np

def design_matrix(x:np.array, degree:int):
    if degree < 1:
        raise ValueError("Invalid value, degree have to be larger or equal to 1.")
    X = np.column_stack((np.ones_like(x), x))
    for i in range(2, degree+1):
        X = np.column_stack((X, x**i))
    return X

def mse(y_true, y_pred):
    n = np.size(y_pred)
    return np.sum((y_true - y_pred)**2) / n

def r2(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    denom = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (num / denom)