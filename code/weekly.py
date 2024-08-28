import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from models import LinRegression


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



if __name__ == '__main__':
    exercise_35_2()
    np.random.seed(2024)
    exercise_35_3()