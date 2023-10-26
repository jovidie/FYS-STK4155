import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from jax import grad


class GradientDecent:

    def __init__(self, x) -> None:
        self.x = 4 + 3*x + x*x
        self.dx = 3 + 2*x


    def func(self):
        return self.x


    def d_func(self):
        return self.dx
    

    def grad_func(self):
        pass
    

    def design_matrix(self, p):
        x = self.x.flatten()
        if p >= 1:
            n = len(x)
            X = np.ones((n, p))
            for i in range(1, p):
                X[:, i] = x**i
            return X
        else: 
            return np.ones_like(x)
        
    
    def learning_schedule(t, t0, t1):
        return t0 / (t+t1)
    

    def solve(self):
        pass
        
        

def func(x):
    return 4 + 3*x + x*x

def d_func(x):
    return 3 + 2*x + np.random.randn(n, 1)

def design_matrix(p, x):
    x = x.ravel()
    if p >= 1:
        n = len(x)
        X = np.ones((n, p))
        for i in range(1, p):
            X[:, i] = x**i
        return X
    else: 
        return np.ones_like(x)
    
def inv_xtx(X, y):
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)


def learning_schedule(t, t0, t1):
    return t0 / (t+t1)

if __name__ == '__main__':
    np.random.seed(2023)
    n = 100
    x = 2*np.random.rand(n, 1)
    y = 4 + 3*x + np.random.randn(n, 1)

    X = design_matrix(2, x)
    XTX = X.T @ X 
    theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y) # inv_xtx(X, y)
    print(f"Theta linreg \n{theta_linreg}")
    hessian = (2/n) * XTX
    eig_val, eig_vec = np.linalg.eig(hessian)
    print(f"Eigenvalue \n{eig_val}")

    theta_gd = np.random.randn(2, 1)
    eta = 1/np.max(eig_val)
    n_iter = 1000

    for i in range(n_iter):
        gradients = 2.0/n * X.T @ ((X @ theta_gd) - y)
        theta_gd = theta_gd - eta*gradients
    
    print(f"Theta GD implementation \n{theta_gd}")

    x_new = np.array([[0], [2]])
    X_new = design_matrix(2, x_new)
    y_predict = X_new @ theta_gd
    y_predict_2 = X_new @ theta_linreg

    n_epoch = 50
    m_batch = 5 # M
    m_size = int(n / m_batch) # m
    t0, t1 = 5, 50

    theta_sdg = np.random.randn(2, 1)

    for e in range(n_epoch):
        for i in range(m_size):
            rand_idx = m_batch * np.random.randint(m_size)
            xi = X[rand_idx: rand_idx+m_batch]
            yi = y[rand_idx: rand_idx+m_batch]
            gradients = (2/m_batch) * xi.T @ ((xi @ theta_sdg) - yi)
            t = e*m_size + i
            eta = learning_schedule(t, t0, t1)
            theta_sdg = theta_sdg - eta*gradients

    print(f"Theta SDG \n{theta_sdg}")
    y_predict_3 = X_new @ theta_sdg

    fig, ax = plt.subplots()
    ax.plot(x_new, y_predict, 'g--', label='GD')
    ax.plot(x_new, y_predict_2, 'b--', label='Linreg')
    ax.plot(x_new, y_predict_3, 'k--', label='SGD')
    ax.plot(x, y, 'ro')
    plt.show()