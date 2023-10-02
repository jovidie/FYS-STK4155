import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Exercise 2
# a)

x = np.random.rand(100, 1)
y = 2.0 + 5*x*x + 0.1*np.random.randn(100, 1)

beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
y_tilde = x @ beta

n = np.size(y_tilde)
mse = np.sum((y - y_tilde).T.dot(y-y_tilde)) / n

fig, ax = plt.subplots()
ax.plot(x, y_tilde, 'k-')
ax.plot(x, y, 'go')
# plt.show()

# b)

model = LinearRegression()
y_model = model.fit(x, y).predict(x)

n_model = np.size(y_model)
mse_model = np.sum((y - y_model).T.dot(y-y_model)) / n