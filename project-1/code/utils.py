import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_plt_params():
    sns.set_theme()
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium"
    }
    plt.rcParams.update(params)

def franke_function(x, y, noise_factor=0):
    """The Franke function is a bivariate test function.

    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        noise_factor (int): adds noise to function, default is 0

    Returns:
        np.ndarray: array of function values
    """
    noise = 0
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    if noise_factor != 0:
        noise = np.random.normal(0, 0.1, len(x)*len(y)) 
        noise = noise_factor.reshape(len(x),len(y))

    return term1 + term2 + term3 + term4 + noise_factor*noise


def design_matrix(x, degree):
    """Create design matrix for polynomial degree n with dimension determined
    by the size of input arrays and degree.

    Args:
        x (np.ndarray): x-values, 1D or 2D array
        y (np.ndarray): y-values, 1D or 2D array
        degree (int): order of polynomial degree
        
    Returns:
        np.ndarray: array with shape (len(x)*len(y), degree)
    """
    p = degree

    if len(x) == 1:
        x = x[0]
        print(x.shape)
        X = np.column_stack((np.ones_like(x), x))
        for i in range(2, p+1):
            X = np.column_stack((X, x**i))
        return X
    
    elif len(x) == 2:
        x, y = x[0], x[1]
        if len(x[0].shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((p + 1) * (p + 2) / 2)

        X = np.zeros((N, l))
        X[:, 0] = np.ones(N)

        for i in range(1, p + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = (x**(i - k)) * (y**k)
        return X

    else:
        raise ValueError(f"{len(x)} is not a valid input dimension, must be 1D or 2D.")


def mse(y_true, y_pred):
    """Calculates the mean squared error.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: mse value
    """
    n = np.size(y_pred)
    return np.sum((y_true - y_pred)**2) / n

def r2(y_true, y_pred):
    """Calculates the R2 score.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: R2 score
    """
    num = np.sum((y_true - y_pred)**2)
    denom = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (num / denom)


if __name__ == '__main__':
    x = np.ones(10) * 2
    y = np.ones(10) * 3
    z = 2.0 + 5*x**2 + x*y + 0.1*np.random.randn(10)

    X = design_matrix(x, y, 5)

    print(X)