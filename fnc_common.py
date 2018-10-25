import numpy as np

def get_unique_2d(data):
	# return unique items from a 2d numpy array
	# input: data = [[a, b], ...]
	# returns a numpy array: [[a, b], ...]
	
	return np.unique(np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * 2)))).view(data.dtype).reshape(-1, 2)
