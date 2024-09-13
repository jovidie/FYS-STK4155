import numpy as np

def design_matrix(x, y, degree):
    """Create design matrix for polynomial degree n with dimension determined
    by the size of input arrays and degree.

    Args:
        x (np.ndarray): x-values, 1D or 2D array
        y (np.ndarray): y-values, 1D or 2D array
        degree (int): order of polynomial degree
        
    Returns:
        np.ndarray: array with shape (len(x)*len(y), degree)
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    # Number of elements in beta
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.zeros((N, l))
    X[:, 0] = np.ones(N)

    for i in range(1, degree + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X


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