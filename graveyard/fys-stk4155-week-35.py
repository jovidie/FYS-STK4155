import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Exercise 1 - Pen and paper proof


# Exercise 2 - Making data

# Dimensions
m, n = 100, 3

# Create array of values for x and y
x = np.random.rand(m, 1)
x.sort(axis=0)
y = 2.0 + 5*x*x + 0.1*np.random.randn(m, 1)

# Create design matrix with p order 2
X = np.zeros((m, n))
X[:, 0] = np.ones((m, 1)).flatten()
X[:, 1] = x.flatten()
X[:, 2] = (x*x).flatten()

# Calculate beta and ỹ using formula in lecture notes
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_tilde = X @ beta

# Place arrays in dataframe, not necessary for further computation
df = pd.DataFrame(X)
df.columns = ['1', 'x', 'x²']
df.insert(3, 'y', y, True)
df.insert(4, 'ỹ', y_tilde, True)

# Create model for fitting data
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_predicted = model.predict(X)

# Calculationg MSE and R² for both ỹ and predicted y, using sklearn
mse_tilde = mean_squared_error(y, y_tilde)
r2_tilde = r2_score(y, y_tilde)
mse_predicted = mean_squared_error(y, y_predicted)
r2_predicted = r2_score(y, y_predicted)

print(f'Computed: MSE={mse_tilde:.6f}, R²={r2_tilde:.6f}')
print(f'SciKit-Learn: MSE={mse_predicted:.6f}, R²={r2_predicted:.6f}')

# Plot computed vs fitted data compared with sklearn
fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(x, y, alpha=0.7, lw=2, label='Experimental')
axes[0].plot(x, y_tilde, alpha=0.7, lw=2, label='Fit')
axes[0].legend()
axes[0].set_title('Computation')
# axes[0].text(0, 0.5, 'Computation', horizontalalignment='center')

axes[1].plot(x, y, alpha=0.7, lw=2, label='Experimental')
axes[1].plot(x, y_predicted, alpha=0.7, lw=2, label='Fit')
axes[1].legend()
axes[1].set_title('SciKit-Learn')

fig.suptitle('Linear Regression')
# fig.text(0.5, 0.02, 'Computation', horizontalalignment='center')
# plt.show()

# Exercise 3
np.random.seed(35)
n = 100
x = np.linspace(-3, 3, n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

# Define feature matrix
X = np.column_stack((np.ones_like(x), x, x**2, x**3, x**4, x**5))

# Split data and find beta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Prediction
# y_tilde = X_train @ beta
# mse_train = mean_squared_error(y_train, y_tilde)
# r2_train = r2_score(y_train, y_tilde)
# print(f"Train \nMSE = {mse_train:.3E} \n R2 = {r2_train:.3E}\n")

# y_predict = X_test @ beta
# mse_test = mean_squared_error(y_test, y_predict)
# r2_test = r2_score(y_test, y_predict)
# print(f"Test \nMSE = {mse_test:.3E} \n R2 = {r2_test:.3E}\n")