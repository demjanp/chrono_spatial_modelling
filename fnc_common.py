import numpy as np

def get_unique_2d(data):
	# return unique items from a 2d numpy array
	
	return np.unique(np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * 2)))).view(data.dtype).reshape(-1, 2)

def running_mean(vals, ts, dt): # TODO obsolete (?)
	# dt = sample interval (years)
	
	dt = dt / 2
	return np.array([vals[(ts >= t - dt) & (ts <= t + dt)].mean() for t in ts])

