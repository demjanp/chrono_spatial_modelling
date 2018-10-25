import numpy as np

from fnc_common import (get_unique_2d)

def get_cost(grid_slope, grid_water, water_limit):
	# calculate cost surface raster
	'''
		Cost function is calculated according to Bevan, A., Lake, M. (2013) Computational Approaches to Archaeological Spaces; chapter: Slope-Dependend Cost Functions
	'''
	# inputs:
	#	grid_slope[i, j] = slope; where i, j are coordinates on raster
	#	grid_water[i, j] = water flow normalized to hectares; where i, j are coordinates on raster
	#	water_limit = limit of water flow which is easily passable
	# returns a numpy array: grid_cost
	#	grid_cost[i, j] = surface cost; where i, j are coordinates on raster
	
	grid_water_c = grid_water.copy()
	
	# add water pixels to prevent spillage
	water_crds = np.argwhere(grid_water_c >= water_limit)
	coords_x = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
	coords_o = np.array([[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]])
	for pnt in water_crds:
		for i in range(coords_x.shape[0]):
			w = grid_water_c[pnt[0], pnt[1]]
			pnt2 = pnt + coords_x[i]
			if (pnt2[0] >= 0) and (pnt2[1] >= 0) and (pnt2[0] < grid_water_c.shape[0]) and (pnt2[1] < grid_water_c.shape[1]) and (grid_water_c[pnt2[0], pnt2[1]] >= water_limit):
				i *= 2
				if i == 0:
					i3 = coords_o.shape[0] - 1
					i4 = 1
				elif i == coords_o.shape[0] - 1:
					i3 = i - 1
					i4 = 0
				else:
					i3 = i - 1
					i4 = i + 1
				pnt3 = pnt + coords_o[i3]
				pnt4 = pnt + coords_o[i4]
				if (pnt3[0] >= 0) and (pnt3[1] >= 0) and (pnt3[0] < grid_water_c.shape[0]) and (pnt3[1] < grid_water_c.shape[1]) and (pnt4[0] >= 0) and (pnt4[1] >= 0) and (pnt4[0] < grid_water_c.shape[0]) and (pnt4[1] < grid_water_c.shape[1]) and (grid_water_c[pnt3[0], pnt3[1]] < water_limit) and (grid_water_c[pnt4[0], pnt4[1]] < water_limit):
					grid_water_c[pnt3[0], pnt3[1]] = w
					grid_water_c[pnt4[0], pnt4[1]] = w
	
	grid_cost = grid_slope.copy()
	grid_cost[grid_cost < 0] = 0
	grid_cost[np.isnan(grid_cost)] = 0
	grid_cost = grid_cost / 100
	grid_cost = 1337.8*grid_cost**6 + 278.19*grid_cost**5 - 517.39*grid_cost**4 - 78.199*grid_cost**3 + 93.419*grid_cost**2 + 19.825*grid_cost + 1.64
	
	# modify cost according to water
	grid_cost[grid_water_c >= water_limit] = np.inf
	
	return grid_cost	

def get_accu_cost(seeds, grid_cost, grid_vert, limits = None):
	# calculate accumulative cost surface raster
	# inputs:
	#	seeds = [[i, j], ...]; where i, j = indexes in grid_cost & grid_vert
	#	grid_cost[i, j] = cost
	#	grid_vert[i, j] = elevation normalized to raster units; where i, j are coordinates on raster
	#	limits = [l, ...]; l = return state after reaching l of generated cells for each l in list
	# returns a numpy array: accu_cost[i,j] = cumulative cost distance (or np.nan if not computed)
	#	if limits is set, return list of matrices in order of limits
	
	coords_o = np.array([[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]])
	dist_o = np.ones(coords_o.shape[0])
	dist_o[::2] = 2
	# coords_o: [[i, j], ...]; relative coordinates around a point in a 2d raster
	# dist_o: [d, ...]; in order of coords_o; squared relative distance to coords around a point in a 2d raster
	
	if not limits is None:
		res = []
		li = 0
	accu_cost = np.zeros(grid_cost.shape, dtype = float)
	accu_cost[:] = np.nan
	for i, j in seeds:
		accu_cost[i,j] = 0
	seeds_collect = seeds[:] if isinstance(seeds, list) else seeds.tolist()
	seed_costs = np.zeros(len(seeds_collect))
	cnt_cells = 0
	while seeds_collect:
		
		idx_seed = np.argmin(seed_costs)
		
		i, j = seeds_collect[idx_seed]
		found = False
		
		# find neighbors of point
		crds = coords_o + [i,j] # coords of free points around i,j
		mask = ((crds[:,0] >= 0) & (crds[:,1] >= 0) & (crds[:,0] < grid_cost.shape[0]) & (crds[:,1] < grid_cost.shape[1]))
		crds = crds[mask]
		if crds.shape[0]:
			dist = dist_o[mask]
			mask = (np.isnan(accu_cost[crds[:,0], crds[:,1]]) & (grid_cost[crds[:,0], crds[:,1]] != np.inf))
			crds = crds[mask]
			dist = dist[mask]
		
		if crds.shape[0]:
		
			# calculate cost & distance to neighbors
			cost1 = grid_cost[i,j]
			cost2 = grid_cost[crds[:,0], crds[:,1]]
			dist = ((grid_vert[i,j] - grid_vert[crds[:,0], crds[:,1]])**2 + dist)**0.5
			
			# calculate cost to travel to each neighbor that adjoins a seed cell
			cost_dist = dist * (cost1 + cost2) / 2
			
			# pick neighbor cell with the least cost
			idxs = np.where(np.abs(cost_dist) != np.inf)[0]
			if idxs.shape[0]:
				
				cost_dist = cost_dist[idxs]
				crds = crds[idxs]
				idx_neigh = np.argmin(cost_dist)
				
				k,l = crds[idx_neigh]
				accu_cost[k,l] = accu_cost[i,j] + cost_dist[idx_neigh]
				
				found = True
				seeds_collect.append([k,l])
				seed_costs = np.hstack((seed_costs, [accu_cost[k,l]]))
				cnt_cells += 1
				if not limits is None:
					if cnt_cells == limits[li]:
						res.append(accu_cost.copy())
						li += 1
						if li == len(limits):
							break
		
		if not found:
			del seeds_collect[idx_seed]
			seed_costs = np.delete(seed_costs, idx_seed)
	
	if limits is None:
		return accu_cost
	else:
		return res

def get_production_areas(coords, dem, cost_surface, vertical_component, production_area):
	# generate production areas around evidence units represented by coordinates based on cost surface
	# inputs:
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	dem[x, y] = elevation a.s.l.; where x, y = coordinates on raster
	#	cost_surface[x, y] = cost; where x, y = coordinates on raster
	#	vertical_component[x, y] = elevation normalized to raster units; where x, y = coordinates on raster
	#	production_area
	# returns a list: production_areas[k] = [[i, j], ...]; where k = index in coords; i, j = indices in cost_surface raster
	
	limit = int(round(production_area * (100 / dem.getCellSize()[0])**2)) # max. number of raster cells per production area
	production_areas = []
	cmax = len(coords)
	c = 1
	for x, y in coords:
		print("\rget_production_areas %d/%d        " % (c, cmax), end = "")
		c += 1
		i, j = dem.getIndex(x, y)
		if i is None:
			production_areas.append(np.array([], dtype = int))
		else:
			res = get_accu_cost([[i,j]], cost_surface, vertical_component, [limit])
			if res:
				production_areas.append(np.argwhere(~np.isnan(res[0])).astype(int))
			else:
				production_areas.append(np.array([], dtype = int))
	return production_areas
	
def find_excluding_pairs(production_areas):
	# find pairs of coordinates where their production areas spatially exclude each other
	# inputs:
	#	production_areas[k] = [[i, j], ...]; where k = index in coords; i, j = indices in cost_surface raster
	# returns numpy array: exclude = [[i1, i2], ...]; where i1, i2 are indices in coords
	
	coords_len = len(production_areas)
	
	extents = [] # find spatial extents of production areas (for optimization reasons)
	for i in range(coords_len):
		if production_areas[i].size:
			extents.append([production_areas[i][:,0].min(), production_areas[i][:,0].max(), production_areas[i][:,1].min(), production_areas[i][:,1].max()])
		else:
			extents.append(None)
	
	done = np.zeros((coords_len, coords_len), dtype = bool)
	exclude = []
	cmax = coords_len**2
	for i1 in range(coords_len):
		for i2 in range(coords_len):
			if (i2 != i1) and (not done[i1, i2]) and (not done[i2, i1]):
				done[i1, i2] = True
				# check if production areas around coordinates i1 and i2 intersect
				if (extents[i1] is not None) and (extents[i2] is not None) and (not ((extents[i1][1] < extents[i2][0]) or (extents[i1][0] > extents[i2][1]) or (extents[i1][3] < extents[i2][2]) or (extents[i1][2] > extents[i2][3]))):
					if np.intersect1d(get_unique_2d(production_areas[i1]), get_unique_2d(production_areas[i2])).shape[0]:
						exclude.append([i1, i2])
	return np.array(exclude, dtype = int)

def find_neighbours(coords, exclude, eu_side):
	# find neighbouring units for each evidence unit represented by its coordinates
	# inputs:
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	exclude = [[i1, i2], ...]; where i1, i2 are indices in coords
	#	eu_side = evidence unit square side (m)
	# returns a list: neighbours[i1] = [i2, ...]; where i1, i2 are indices in coords
	
	neighbours = []
	for i in range(coords.shape[0]):
		neighbours.append(np.where((np.abs(coords[i] - coords) <= [2.2 * eu_side, 1.1 * eu_side]).all(axis = 1))[0].astype(int).tolist())
	
	# remove coords that do not exclude each other from neighbours (they are across water and cannot form clusters)
	collect = []
	for i1 in range(coords.shape[0]):
		collect.append([])
		for i2 in neighbours[i1]:
			if (i1 == i2) or ([i1, i2] in exclude) or ([i2, i1] in exclude):
				collect[i1].append(i2)
	neighbours = collect
	return neighbours

