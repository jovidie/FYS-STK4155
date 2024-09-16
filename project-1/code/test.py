import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils import set_plt_params, design_matrix, franke_function, mse, r2
from models import LinearRegression, RidgeRegression, LassoRegression


def test_franke_function():
    set_plt_params()
    x1 = np.arange(0, 1, 0.05)
    x2 = np.arange(0, 1, 0.05)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x1, x2 = np.meshgrid(x1, x2)
    y = franke_function(x1, x2)

    surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def test_linear_regression():
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y = 2.0 + 5*x1**2 + x1*x2 + 0.1*np.random.randn(100)
    degrees = np.linspace(1, 15)

    mse_history = []
    r2_history = []

    for p in degrees:
        X = design_matrix([x1, x2], p)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse_history.append(mean_squared_error(y_test, y_pred))
        r2_history.append(r2_score(y_test, y_pred))

    fig, ax = plt.subplots()
    ax.plot(degrees, mse_history, label="MSE")
    ax.plot(degrees, r2_history, label="R2 score")
    ax.set_yscale("log")
    ax.legend()
    plt.show()


def test_ridge_regression():
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y = franke_function(x1, x2)
    degrees = np.arange(1, 15)
    lmbdas = [0.0001, 0.001, 0.01, 0.1, 1.0]

    mse_history = np.zeros((len(lmbdas), len(degrees)))
    r2_history = np.zeros((len(lmbdas), len(degrees)))

    for i, p in enumerate(degrees):
        X = design_matrix([x1, x2], p)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for j, lmbda in enumerate(lmbdas):
            model = RidgeRegression(lmbda)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse_history[j, i] = mean_squared_error(y_test, y_pred)
            r2_history[j, i] = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for j in range(len(lmbdas)):
        ax[0].plot(degrees, mse_history[j, :], label=rf"$\lambda = {lmbdas[j]}$")
        ax[1].plot(degrees, r2_history[j, :], label=rf"$\lambda = {lmbdas[j]}$")

    ax[0].set_title("MSE")
    ax[0].legend()
    ax[1].set_title("R2 score")
    ax[1].legend()
    plt.show()


def test_lasso_regression():
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y = franke_function(x1, x2)
    degrees = np.arange(1, 15)
    lmbdas = [0.0001, 0.001, 0.01, 0.1, 1.0]

    mse_history = np.zeros((len(lmbdas), len(degrees)))
    r2_history = np.zeros((len(lmbdas), len(degrees)))

    for i, p in enumerate(degrees):
        X = design_matrix([x1, x2], p)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for j, lmbda in enumerate(lmbdas):
            model = LassoRegression(lmbda)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse_history[j, i] = mean_squared_error(y_test, y_pred)
            r2_history[j, i] = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for j in range(len(lmbdas)):
        ax[0].plot(degrees, mse_history[j, :], label=rf"$\lambda = {lmbdas[j]}$")
        ax[1].plot(degrees, r2_history[j, :], label=rf"$\lambda = {lmbdas[j]}$")

    ax[0].set_title("MSE")
    ax[0].legend()
    ax[1].set_title("R2 score")
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    # test_franke_function()
    # test_linear_regression()
    # test_ridge_regression()
    test_lasso_regression()