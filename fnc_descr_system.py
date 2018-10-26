import numpy as np

from fnc_common import (get_unique_2d)
from fnc_data import (GeoTIFF)
from fnc_stats_spatial import (get_cost, find_excluding_pairs, find_neighbours)
from fnc_stats_temporal import (get_dating_intervals, get_chronological_phases, get_intervals_per_coords)


def get_descriptive_system(data, interval_thresh, eu_side, production_area, water_limit, path_dem, path_slope, path_water):
	# create a descriptive system based on evidence
	# inputs:
	#	data = [[BP_from, BP_to, X, Y], ...]; where BP_from, BP_to = dating interval in calendar years BP; X, Y = coordinates of evidence unit
	#	interval_thresh = threshold for a time interval lenght under which this interval is considered contemporary with another if it overlaps with it by any length
	#	eu_side = evidence unit square side (m)
	#	production_area = size of production area (ha)
	#	water_limit = limit of water flow which is easily passable (= amount of hectares from which the water has accumulated in the flow accumulation model)
	#	path_dem = path in string format to file containing the DEM raster in GeoTIFF format
	#	path_slope = path in string format to file containing the Slope raster in GeoTIFF format
	#	path_water = path in string format to file containing the Water raster in GeoTIFF format
	# returns: coords, intervals, phases_chrono, intervals_coords, neighbours, exclude, production_areas, extent, raster_shape
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	intervals = [[BP_from, BP_to], ...]
	#	phases_chrono[pi] = [[i, ...], ...]; chronological phases; where pi = index of phase and i = index in intervals
	#	intervals_coords[i] = [[BP_from, BP_to], ...]; where i = index in coords
	#	neighbours[i1] = [i2, ...]; where i1, i2 are indices in coords
	#	exclude = [[i1, i2], ...]; where i1, i2 are indices in coords
	#	production_areas[k] = [[i, j], ...]; where k = index in coords; i, j = indices in cost_surface raster
	#	extent = [xmin, xmax, ymin, ymax]; where xmin, xmax, ymin, ymax are spatial limits of the examined area
	#	raster_shape = (rows, columns)

	# reduce resolution of data coordinates to eu_side
	data[:, 2:] = (np.round(data[:, 2:] / eu_side) * eu_side)

	coords = get_unique_2d(data[:, 2:])  # [[X, Y], ...]

	# load DEM, Slope and Water rasters
	dem = GeoTIFF(path_dem)  # Digital elevation model
	slope = GeoTIFF(path_slope).get_raster()
	water = GeoTIFF(path_water)

	xmin, ymin, xmax, ymax = dem.get_extent()
	extent = [xmin, xmax, ymin, ymax]
	raster_shape = slope.shape
	vertical_component = dem.get_raster()  # DEM normalized to raster units; Matrix [x, y] = height in raster units
	vertical_component[np.isnan(vertical_component)] = 0
	vertical_component /= dem.get_cell_size()[0]
	water_cellsize = water.get_cell_size()[0]
	water = water.get_raster()  # water[x, y] = water flow
	water[water < 0] = 0
	water = (water * (water_cellsize ** 2)) / (100 ** 2)  # convert to hectares
	cost_surface = get_cost(slope, water, water_limit)  # cost_surface[x, y] = cost

	intervals = get_dating_intervals(data)  # [[BP_from, BP_to], ...]

	phases_chrono, contemporary = get_chronological_phases(intervals, interval_thresh)  # phases_chrono[pi] = [[i, ...], ...]; contemporary[i1] = [i2, ...]

	intervals_coords = get_intervals_per_coords(data, coords)  # intervals_coords[i] = [[BP_from, BP_to], ...]; where i = index in coords

	#	production_areas = get_production_areas(coords, dem, cost_surface, vertical_component, production_area)
	import json  # DEBUG
	#	with open("tmp_pa.json", "w") as f: # DEBUG
	#		json.dump([area.tolist() for area in production_areas], f) # DEBUG
	with open("tmp_pa.json", "r") as f:  # DEBUG
		production_areas = [np.array(area, dtype=int) for area in json.load(f)]  # DEBUG

	# filter out coords where a production area cannot be modelled (e.g. they lie in a riverbed)
	idxs = [i for i in range(len(coords)) if len(production_areas[i]) > 0]
	intervals_coords = [intervals_coords[i] for i in idxs]
	production_areas = [production_areas[i] for i in idxs]
	coords = coords[idxs]

	exclude = find_excluding_pairs(production_areas)  # [[i1, i2], ...]

	neighbours = find_neighbours(coords, exclude, eu_side)  # neighbours[i1] = [i2, ...]

	return coords, intervals, phases_chrono, intervals_coords, neighbours, exclude, production_areas, extent, raster_shape
