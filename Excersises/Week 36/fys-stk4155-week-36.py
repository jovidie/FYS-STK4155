import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Higher order model
def poly_feature_matrix(p: int, x: np.ndarray, scale=False) -> np.ndarray:
    if p < 1:
        print(f"n = {n} is not valid polynomial order")
        return
    X = np.column_stack((np.ones_like(x), x))
    for i in range(2, p+1):
        X = np.column_stack((X, x**i))
    if scale is True:
        X = (X - np.mean(X))
    return X 

# General computation of beta
def compute_beta(X: np.ndarray, y: np.ndarray, lmbda: List=None) -> np.ndarray:
    beta = []
    if lmbda is None:
        # beta_ols
        A = np.linalg.inv(X.T @ X) @ X.T
        return A @ y 
    else:
        # beta_ridge
        XX = X.T @ X
        p, _ = XX.shape
        for l in lmbda:
            I = np.identity(p) * l 
            A = np.linalg.inv(XX + I) @ X.T
            beta.append(A @ y) 
        return beta

np.random.seed(35)
n = 100
p = 15
x = np.linspace(-3, 3, n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
X = poly_feature_matrix(p, x)

# Split data and find beta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0]
beta_ridge = compute_beta(X=X_train, y=y_train, lmbda=lmbda)

# Prediction Ridge
y_tilde = []
y_predict = []
mse_train = []
r2_train = []
mse_test = []
r2_test = []

for b in beta_ridge:
    y_tilde.append(X_train @ b)
    y_predict.append(X_test @ b)

print("Ridge Regression")
for i in range(len(y_tilde)):
    print(f"Lambda: {lmbda[i]}")
    mse_train.append(mean_squared_error(y_train, y_tilde[i]))
    r2_train.append(r2_score(y_train, y_tilde[i]))
    print(f"Train \nMSE = {mse_train[i]} \n R2 = {r2_train[i]}")

    mse_test.append(mean_squared_error(y_test, y_predict[i]))
    r2_test.append(r2_score(y_test, y_predict[i]))
    print(f"Test \nMSE = {mse_test[i]} \n R2 = {r2_test[i]}\n")

beta_ols = compute_beta(X=X_train, y=y_train)

# Prediction OLS
y_tilde = X_train @ beta_ols
y_predict = X_test @ beta_ols

print("OLS")
mse_train = mean_squared_error(y_train, y_tilde)
r2_train = r2_score(y_train, y_tilde)
print(f"Train \nMSE = {mse_train} \n R2 = {r2_train}")

mse_test = mean_squared_error(y_test, y_predict)
r2_test = r2_score(y_test, y_predict)
print(f"Test \nMSE = {mse_test} \n R2 = {r2_test}\n")