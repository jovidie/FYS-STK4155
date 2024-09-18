from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from models import LinRegression, LinearReg

def design_matrix(x, degree):
    p = degree


    X = np.column_stack((np.ones_like(x), x))

    if degree < 2:
        return X
    else:
        for i in range(2, p+1):
            X = np.column_stack((X, x**i))
        return X

def mse(y_true, y_pred):
    n = np.size(y_pred)
    return np.sum((y_true - y_pred)**2) / n

def r2(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    denom = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (num / denom)

def exercise_35_2():
    x = np.random.rand(100)
    y = 2.0 + 5*x**2 + 0.1*np.random.randn(100)

    # Create design matrix 
    X = np.zeros((len(x), 3))
    X[:, 0] = 1
    X[:, 1] = x 
    X[:, 2] = x*x 

    beta = np.linalg.inv(X.T @ X) @ X.T @ y 
    y_pred = X @ beta

    poly = PolynomialFeatures(degree=2)
    X_model = poly.fit_transform(x[:, np.newaxis])
    model = LinearRegression()
    model.fit(X, y)
    model_pred = model.predict(X_model)

    mse_man = mean_squared_error(y, y_pred)
    r2_man = r2_score(y, y_pred)
    mse_sk = mean_squared_error(y, model_pred)
    r2_sk = r2_score(y, model_pred)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b.', label="Data")
    ax.plot(x, y_pred, 'ro', label=f"Own code: MSE = {mse_man:.4f}, R2 = {r2_man:.4f}")
    ax.plot(x, model_pred, 'gx', label=f"SciKit: MSE = {mse_sk:.4f}, R2 = {r2_sk:.4f}")
    ax.legend()
    fig.savefig("../latex/figures/week35_ex2.pdf")
    # plt.show()


def exercise_35_3():
    n = 100
    
    x = np.linspace(-3, 3, n)
    # x = x.reshape(-1, 1)
    noise = np.random.normal(0, 0.1, x.shape)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + noise 

    # poly = PolynomialFeatures(degree=4)
    # X = poly.fit_transform(x[:, np.newaxis])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    mse_history = []
    r2_history = []

    for degree in range(15):
        model = LinRegression(degree)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse_loss, r2_val = model.compute_error(y_test, y_pred)
        mse_history.append(mse_loss)
        r2_history.append(r2_val)

    d = np.arange(15)
    p_optim = np.argmin(mse_history)
    fig, ax = plt.subplots()
    ax.plot(d, mse_history, label="MSE")
    ax.plot(d, r2_history, label="R2")
    ax.legend()
    ax.set_xlabel(f"Optimal MSE = {mse_history[p_optim]:.4f}, polynomial degree = {np.argmin(mse_history)}")
    fig.savefig("../latex/figures/week35_ex3.pdf")
    # plt.show()


def exercise_36_2():
    n = 100
    lmbdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    x = np.linspace(-3, 3, n)
    noise = np.random.normal(0, 0.1, x.shape)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + noise 

    # Scale data as standard scaling, and then split into train test

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    mse_history = []
    r2_history = []

    for degree in range(15):
        model = LinRegression(degree)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse_loss, r2_val = model.compute_error(y_test, y_pred)
        mse_history.append(mse_loss)
        r2_history.append(r2_val)


def exercise_38():
    n = 40
    degree = 14
    n_bootstraps = 100

    x = np.random.rand(100) # .reshape(-1, 1)
    noise = np.random.normal(0, 0.1, x.shape)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + noise 

    # error_train = np.empty(degree)
    error_test = np.empty(degree)
    bias = np.empty(degree)
    variance = np.empty(degree)

    for p in range(1, degree):

        X = design_matrix(x, p)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        y_tilde = np.empty((y_train.shape[0], n_bootstraps))
        y_pred = np.empty((y_test.shape[0], n_bootstraps))

        # mse_train = np.empty(n_bootstraps)
        mse_test = np.empty(n_bootstraps)

        for i in range(n_bootstraps):
            X_, y_ = resample(X_train, y_train)
            model = LinearRegression()
            model.fit(X_, y_)

            y_tilde[:, i] = model.predict(X_train).ravel()
            y_pred[:, i] = model.predict(X_test).ravel()

            # mse_train[i] = mse(y_, y_tilde[:, i])
            mse_test[i] = mse(y_test, y_pred[:, i])

        # error_train[p] = np.mean(mse_train)
        # error_test[p] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        # bias[p] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        # variance[p] = np.mean(np.var(y_pred, axis=1, keepdims=True))
        # error_train[p] = np.mean(mse_train)
        error_test[p] = np.mean(mse_test)
        bias[p] = np.mean((y_test - np.mean(y_pred, axis=1))**2)
        variance[p] = np.mean(np.var(y_pred, axis=1, keepdims=True))

    # return (error_train, error_test, bias, variance)
    return (error_test, bias, variance)


if __name__ == '__main__':
    # exercise_35_2()
    np.random.seed(2024)
    # exercise_35_3()
    error_test, bias, variance = exercise_38()

    degrees = np.arange(14)
    fig, ax = plt.subplots()
    # ax.plot(degrees, error_train, label=r"$\epsilon_{train}$")
    ax.plot(degrees, error_test, label=r"$\epsilon_{test}$")
    ax.plot(degrees, bias, label="Bias")
    ax.plot(degrees, variance, label="Variance")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("../latex/figures/ex38.pdf")