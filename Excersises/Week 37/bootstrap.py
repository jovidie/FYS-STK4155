import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import sys 

sys.path.append('/home/jovie/Documents/Studies/FYS-STK4155/Assignments/FYS_STK_Project_1/code/')
from part_a import Franke_function, design_matrix

np.random.seed(12)

n = 100
n_boostraps = 100
maxdegree = 16


# Make data set.
x_vec = np.linspace(0, 1, n)
y_vec = np.linspace(0, 1, n)
# y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
x, y = np.meshgrid(x_vec, y_vec)
franke = Franke_function(x, y, noise=True)

# X = np.column_stack((x.reshape(-1), y.reshape(-1)))
F = franke.reshape(-1, 1)

error_train = np.zeros(maxdegree)
error_test = np.zeros(maxdegree)
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
# x_train, x_test, y_train, y_test = train_test_split(X, F, test_size=0.2)



for degree in range(maxdegree):
    # poly = PolynomialFeatures(degree=degree)
    # M = poly.fit_transform(X)
    X = design_matrix(x, y, degree)
    x_train, x_test, y_train, y_test = train_test_split(X, F, test_size=0.2)
    # model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    model = LinearRegression(fit_intercept=False)
    # y_tilde = np.empty((y_train.shape[0], n_boostraps))
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    # y_pred = np.zeros(n_boostraps)
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)
        clf = model.fit(x_, y_)
        # y_tilde[:, i] = clf.predict(x_train).ravel()
        # y_pred[:, i] = clf.predict(x_test).ravel()
        # y_tilde[:, i] = model.fit(x_, y_).predict(x_train).ravel()
        # y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
        y_tilde = clf.predict(x_train)#.ravel()
        y_predict = clf.predict(x_test)#.ravel()
        y_pred[:, i] = y_predict.ravel()
        error_train[degree] += mean_squared_error(y_train, y_tilde)
        error_test[degree] += mean_squared_error(y_test, y_predict)
        # bias[degree] += (y_test - np.mean(y_predict, axis=1, keepdims=True))**2
        # variance[degree] += np.var(y_predict, axis=1, keepdims=True)

    polydegree[degree] = degree
    error_train[degree] /= n_boostraps
    error_test[degree] /= n_boostraps
    # bias[degree] /= n_boostraps 
    # variance[degree] /= n_boostraps
    
    # error_train[degree] = np.mean( np.mean((y_train - y_tilde)**2, axis=1, keepdims=True) )
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    # bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    # variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
#     # print('Polynomial degree:', degree)
#     # print('Error train:', error_train[degree])
#     # print('Error test:', error[degree])
#     # print('Bias^2:', bias[degree])
#     # print('Var:', variance[degree])
#     # print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

fig, ax = plt.subplots()
ax.plot(polydegree, error_train, label='Train')
ax.plot(polydegree, error_test, label='Test')
ax.plot(polydegree, error, label="Test np.mean")
# ax.plot(polydegree, np.log10(error_train), label='Train')
# ax.plot(polydegree, np.log10(error_test), label='Test')
# ax.plot(polydegree, np.log10(error), label="Test np.mean")
# plt.plot(polydegree, bias, label='bias')
# plt.plot(polydegree, variance, label='Variance')
ax.set_xlabel('Model Complexity')
ax.set_ylabel('Prediction Error')
# ax.plot(polydegree, bias, label='Bias')
# ax.plot(polydegree, variance, label='Variance')
ax.set_xlim((0, 15))
# ax.set_ylim(())
ax.legend()
plt.show()