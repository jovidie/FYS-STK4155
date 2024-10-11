import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_plt_params():
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium", 
        "savefig.dpi": 300
    }
    plt.rcParams.update(params)
    

def franke_function(x1, x2, noise_var=0):
    """The Franke function is a bivariate test function.

    Args:
        x1 (np.ndarray): x1-values
        x2 (np.ndarray): x2-values
        noise_var (float): add noise with variation [0, 1]

    Returns:
        np.ndarray: array of function values
    """
    m, n = len(x1), len(x2)
    noise = np.random.normal(0, noise_var, n*m).reshape(m, n)
    
    term1 = 0.75*np.exp(-(0.25*(9*x1-2)**2) - 0.25*((9*x2-2)**2))
    term2 = 0.75*np.exp(-((9*x1+1)**2)/49.0 - 0.1*(9*x2+1))
    term3 = 0.5*np.exp(-(9*x1-7)**2/4.0 - 0.25*((9*x2-3)**2))
    term4 = -0.2*np.exp(-(9*x1-4)**2 - (9*x2-7)**2)

    franke = term1 + term2 + term3 + term4

    if noise_var != 0:
        return franke + noise
    
    return franke


def design_matrix(x1, x2, p):
    """Create design matrix for polynomial degree n with dimension determined
    by the size of input arrays and degree.

    Args:
        x1 (np.ndarray): x1-values, 1D or 2D array
		x2 (np.ndarray): x2-values, 1D or 2D array
        p (int): order of polynomial degree
        
    Returns:
        np.ndarray: array with polynomial features
    """
    if len(x1.shape) > 1:
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)

    m = len(x1)
	# Number of elements in beta
    n = int((p+1)*(p+2)/2)

    X = np.ones((m, n))

    for i in range(1, p+1):
        q = int((i)*(i+1)/2)
        for j in range(i + 1):
            X[:, q+j] = (x1**(i-j))*(x2**j)

    return X


def mse(y_true, y_pred):
	"""Calculate the mean square error of the fit.
	
	Args:
	    y_true (np.ndarray): data to fit
		model (np.ndarray): predicted model
		
	Returns:
	    np.ndarray: mean square error of fit
	"""
	return np.sum((y_true - y_pred)**2) / len(y_true)


def r2(y_true, y_pred):
    """Calculate the R2 score of the fit

    Args:
        data (np.ndarray): original data to fit
        model (np.ndarray): predicted model
		
    Returns:
        np.ndarray: R2 score of fit
    """
    num = np.sum((y_true - y_pred)**2)
    denom = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (num / denom)

