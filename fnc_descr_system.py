import numpy as np
from fnc_data import (GeoTIFF)
from fnc_stats_spatial import (get_cost, get_production_areas, find_excluding_pairs, find_neighbours)
from fnc_stats_temporal import (get_dating_intervals, get_chronological_phases, get_intervals_per_coords)
from fnc_common import (get_unique_2d)

def get_descriptive_system(data, interval_thresh, eu_side, production_area, water_limit, path_dem, path_slope, path_water):
	
	# reduce resolution of data coordinates to eu_side
	data[:,2:] = (np.round(data[:,2:] / eu_side) * eu_side)
	
	coords = get_unique_2d(data) # [[X, Y], ...]

	# load DEM, Slope and Water rasters
	dem = GeoTIFF(path_dem) # Digital elevation model
	slope = GeoTIFF(path_slope).getRaster()
	water = GeoTIFF(path_water)

	xmin, ymin, xmax, ymax = dem.getExtent()
	extent = [xmin, xmax, ymin, ymax]
	raster_shape = slope.shape
	vertical_component = dem.getRaster() # DEM normalized to raster units; Matrix [x, y] = height in raster units
	vertical_component[np.isnan(vertical_component)] = 0
	vertical_component /= dem.getCellSize()[0]
	water_cellsize = water.getCellSize()[0]
	water = water.getRaster() # Matrix [x, y] = water flow
	water[water < 0] = 0
	water = (water * (water_cellsize**2)) / (100**2) # convert to hectares
	cost_surface = get_cost(slope, water, water_limit) # Matrix [x, y] = cost

	intervals = get_dating_intervals(data) # [[BP_from, BP_to], ...]

	phases_chrono, contemporary = get_chronological_phases(intervals, interval_thresh) # phases_chrono[pi] = [[i, ...], ...]; contemporary[i1] = [i2, ...]

	intervals_coords = get_intervals_per_coords(data, coords) # intervals_coords[i] = [[BP_from, BP_to], ...]; where i = index in coords
	
	'''
	production_areas = get_production_areas(coords, dem, cost_surface, vertical_component, production_area)
	''' # DEBUG
	import json # DEBUG
#	with open("tmp_pa_br_ha.json", "w") as f: # DEBUG
#		json.dump([area.tolist() for area in production_areas], f) # DEBUG
	with open("tmp_pa_br_ha.json", "r") as f: # DEBUG
		production_areas = [np.array(area, dtype = int) for area in json.load(f)] # DEBUG

	
	# filter out coords where a production area cannot be modelled (e.g. they lie in a riverbed)
	idxs = [i for i in range(len(coords)) if len(production_areas[i]) > 0]
	intervals_coords = [intervals_coords[i] for i in idxs]
	production_areas = [production_areas[i] for i in idxs]
	coords = coords[idxs]
	
	exclude = find_excluding_pairs(production_areas) # [[i1, i2], ...]
	
	neighbours = find_neighbours(coords, exclude, eu_side) # neighbours[i1] = [i2, ...]
	
	return coords, intervals, phases_chrono, intervals_coords, neighbours, exclude, production_areas, extent, raster_shape

