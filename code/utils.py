import numpy as np

def mse(y_true, y_pred):
    n = np.size(y_pred)
    return np.sum((y_true - y_pred)**2) / n

def r2(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    denom = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (num / denom)