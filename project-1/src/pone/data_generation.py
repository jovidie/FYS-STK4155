import numpy as np
from imageio.v3 import imread

from pone.utils import franke_function

def create_terrain_data(filename, N, scaled=False):
	"""Load terrain data from file and create input and output arrays.
	
	Args:
        filename (str): filename to import data from
		N (int): number of rows and columns to use from terrain
		scaled (bool): returns the data scaled, default is False
		
	Returns:
        tuple: arrays of input and terrain data
	"""
	data = f"data/{filename}"
	terrain = imread(data)

	# Create y of given data points
	y = np.zeros((N, N))
	for i in range(N):
		for j in range(N):
			y[i, j] = terrain[i, j]

	# Create x1 and x2
	x1_ = np.sort(np.random.uniform(0, 1, N))
	x2_ = np.sort(np.random.uniform(0, 1, N))
	x1, x2= np.meshgrid(x1_, x2_)

	if scaled:
		y = y.ravel()
		y_scaled = (y - y.mean()) / y.std()
		return (x1, x2, y_scaled.reshape(-1, N))
	
	else:
		return (x1, x2, y)



def create_function_data(N, noise_var=0, scaled=False):
	"""Create input and output data using the Franke function
	
	Args:
        N (int): number of data points
	    noise_factor (float): add noise to function, default is 0
		scaled (bool): returns the data scaled, default is False
		
	Returns:
	    tuple: arrays of input and function data
	"""
	# Create x and y
	x1_ = np.sort(np.random.uniform(0, 1, N))
	x2_ = np.sort(np.random.uniform(0, 1, N))
	x1, x2= np.meshgrid(x1_, x2_)
	
	# x1, x2 = x1.ravel(), x2.ravel()
	y = franke_function(x1, x2, noise_var)

	if scaled:
		y = y.ravel()
		y_scaled = (y - y.mean()) / y.std()
		return (x1, x2, y_scaled.reshape(-1, N))
	
	else:
		return (x1, x2, y)
	

