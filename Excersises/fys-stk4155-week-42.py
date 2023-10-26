import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import jax.numpy as jnp
# from jax import grad


class StochasticGradientDecent:
    """Doc"""

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.x = 2*np.random.rand(n, 1)
        self.y = 4 + 3*self.x + np.random.randn(n, 1)
        self.X = None
        self._eig = None
        self._theta = None
        self._predict = None
        # self.x = input[0]
        # self.y = input[1]


    def eta(self):
        xtx = self.X.T @ self.X
        hessian = (2/self.n) * xtx
        self.eig, _ = np.linalg.eig(hessian)
        self.eta = 1/np.max(self.eig)


    def design_matrix(self):
        x = np.ravel(self.x)
        n = len(x)
        if self.p >= 1:
            X = np.ones((n, self.p))
            for i in range(1, self.p):
                X[:, i] = x**i
            self.X = X
        else:
            self.X = np.ones((n, 1))



    def learning_schedule(self, t, t0, t1):
        return t0 / (t+t1)

    
    def theta(self, n_epoch, m_batch, t0, t1):
        m_size = int(self.n / m_batch)
        theta = np.random.randn(self.p, 1)

        for e in range(n_epoch):
            for i in range(m_size):
                rand_idx = m_batch * np.random.randint(m_size)
                Xi = self.X[rand_idx: rand_idx+m_batch]
                yi = self.y[rand_idx: rand_idx+m_batch]
                gradients = (2/m_batch) * Xi.T @ ((Xi @ theta) - yi)
                t = e*m_size + i
                eta = self.learning_schedule(t, t0, t1)
                theta = theta - eta*gradients
        self.theta = theta

    
    def solve(self):
        x_test = np.random.rand(self.n, self.p)
        self.predict = x_test @ self.theta
        return self.predict
    


if __name__ == '__main__':
    model = StochasticGradientDecent(100, 2)
    model.design_matrix()
    model.theta(50, 5, 5, 50)
    y_predict = model.solve()
    print(y_predict)