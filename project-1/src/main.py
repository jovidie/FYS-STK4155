import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pone.utils import franke_function
from pone.models import LinearRegression
from pone.resamplers import Bootstrapper

def create_function_data(n, noise_factor=0):
	"""Create input data, and output data using the Franke function
	
	Args:
        n (int): number of data points
	    noise_factor (int): add noise to function, default is 0
		
	Returns:
	    tuple: arrays of input and output data
	"""
	x_ = np.sort(np.random.uniform(0, 1, n))
	y_ = np.sort(np.random.uniform(0, 1, n))
	x, y = np.meshgrid(x_, y_)
	
	x, y = x.ravel(), y.ravel()
	z = franke_function(x, y, noise_factor)
	
	return (x, y, z.ravel())

def main():
    x, y, z = create_function_data(100, 0.1)
    model = LinearRegression()
    