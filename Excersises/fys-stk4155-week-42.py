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
        self._x = np.linspace(0, 1, n).reshape(-1, 1)
        self._y = 4 + 3*self._x + np.random.randn(n, 1)
        self._X = None
        self._eig = None
        self._theta = None
        self._predict = None
        # self.x = input[0]
        # self.y = input[1]


    def get_x(self):
        return self._x
    

    def get_y(self):
        return self._y
    

    def eta(self):
        xtx = self._X.T @ self._X
        hessian = (2/self.n) * xtx
        self.eig, _ = np.linalg.eig(hessian)
        self._eta = 1/np.max(self.eig)


    def design_matrix(self):
        x = np.ravel(self._x)
        n = len(x)
        if self.p >= 1:
            X = np.ones((n, self.p))
            for i in range(1, self.p):
                X[:, i] = x**i
            self._X = X
        else:
            self._X = np.ones((n, 1))



    def learning_schedule(self, t, t0, t1):
        return t0 / (t+t1)

    
    def theta(self, n_epoch, batch_size, t0, t1):
        n_batch = int(self.n / batch_size)
        theta = np.random.randn(self.p, 1)

        for e in range(n_epoch):
            for i in range(n_batch):
                rand_idx = batch_size * np.random.randint(n_batch)
                Xi = self._X[rand_idx: rand_idx+batch_size]
                yi = self._y[rand_idx: rand_idx+batch_size]
                gradients = (2/batch_size) * Xi.T @ ((Xi @ theta) - yi)
                t = e*n_batch + i
                eta = self.learning_schedule(t, t0, t1)
                theta = theta - eta*gradients
        self._theta = theta

    
    def solve(self):
        X_new = np.random.rand(self.n, self.p)
        self._predict = X_new.dot(self._theta)
        return self._predict
    

def main():
    model = StochasticGradientDecent(100, 2)
    model.design_matrix()
    model.theta(50, 5, 5, 50)
    y_predict = model.solve()

    x = model.get_x()
    y_true = model.get_y()

    fig, ax = plt.subplots()
    ax.scatter(x, y_true, color="black")
    ax.scatter(x, y_predict, color="seagreen")
    plt.show()


if __name__ == '__main__':
    main()