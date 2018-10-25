import numpy as np

from fnc_chrono_spatial_modelling import (find_clusters)
from fnc_common import (get_unique_2d)

def generate_production_area_maps(solutions, raster_shape, neighbours, production_areas):
	# returns a numpy array: pa_grids[pi, i, j] = p; where pi = index of phase; i, j = indices in 2D raster with cell size = EU_SIDE; p = probability of presence of production area
	
	phases_n = solutions.shape[2]
	
	pa_grids = np.zeros((phases_n, raster_shape[0], raster_shape[1]))
	for pi in range(phases_n):
		for solution in solutions:
			idxs = np.where(solution[:,pi])[0]
			clusters = find_clusters(idxs, neighbours)
			for cluster in clusters:
				collect = []
				for i in cluster:
					collect += production_areas[i].tolist()
				collect = get_unique_2d(np.array(collect, dtype = int))
				pa_grids[pi, collect[:,0], collect[:,1]] += 1
		pa_grids[pi] /= solutions.shape[0]
	
	return pa_grids

