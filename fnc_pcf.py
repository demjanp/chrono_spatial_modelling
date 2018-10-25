import numpy as np

from fnc_common import (get_unique_2d)
from fnc_data import (load_examined_coords)


def calculate_PCF(coords, r_max, eu_side):
	# calculate Pair Correlation Function (PCF) for evidence units represented by their coordinates
	"""
		Compute the two-dimensional pair correlation function, also known
		as the radial distribution function, for a set of circular particles
		contained in a square region of a plane.  This simple function finds
		reference particles such that a circle of radius r_max drawn around the
		particle will fit entirely within the square, eliminating the need to
		compensate for edge effects.  If no such particles exist, an error is
		returned
		
		Modified after Craig Finch (https://github.com/cfinch/Shocksolution_Examples/tree/master/PairCorrelation)
	"""
	# inputs:
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	r_max = maximum radius to calculate for
	#	eu_side = evidence unit square side (m)
	# returns a numpy array: pcf = [[r, g], ...]; where r = radius of the annulus used to compute g(r), g = average correlation function g(r) based on all solutions

	sx = coords[:, 0].max() - coords[:, 0].min()
	sy = coords[:, 1].max() - coords[:, 1].min()
	if r_max is None:
		r_max = min(sx, sy) / 4

	# Number of particles in ring/area of ring/number of reference particles/number density
	# area of ring = pi*(r_outer**2 - r_inner**2)

	edges = np.arange(0., r_max + 1.1 * eu_side, eu_side)
	num_increments = len(edges) - 1
	radii = np.zeros(num_increments)

	for i in range(num_increments):
		radii[i] = (edges[i] + edges[i + 1]) / 2.
		r_outer = edges[i + 1]
		r_inner = edges[i]

	x, y = (coords - coords.min(axis=0)).T

	# Find particles which are close enough to the box center that a circle of radius
	# r_max will not cross any edge of the box
	bools1 = x > r_max
	bools2 = x < (sx - r_max)
	bools3 = y > r_max
	bools4 = y < (sy - r_max)
	interior_indices, = np.where(bools1 * bools2 * bools3 * bools4)
	num_interior_particles = len(interior_indices)

	if num_interior_particles < 1:
		return None

	g = np.zeros([num_interior_particles, num_increments])
	number_density = len(x) / (sx * sy)

	# Compute pairwise correlation for each interior particle
	for p in range(num_interior_particles):
		index = interior_indices[p]
		d = np.sqrt((x[index] - x) ** 2 + (y[index] - y) ** 2)
		d[index] = 2 * r_max
		d[np.isnan(d)] = 0

		(result, bins) = np.histogram(d, bins=edges, normed=False)
		g[p, :] = result / number_density

	# Average g(r) for all interior particles and compute radii
	g_average = np.zeros(num_increments)
	for i in range(num_increments):
		g_average[i] = np.mean(g[:, i]) / (np.pi * (r_outer ** 2 - r_inner ** 2))

	return np.vstack((radii, g_average)).T


def calculate_PCF_solutions(solutions, coords, eu_side):
	# calculate PCF for the whole set of solutions
	# inputs:
	#	solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	eu_side = evidence unit square side (m)
	# returns a numpy array: pcf[pi] = [r, g]; where pi = index of phase, r = radius of the annulus used to compute g(r), g = average correlation function g(r)

	"""
		Modified after Craig Finch (https://github.com/cfinch/Shocksolution_Examples/tree/master/PairCorrelation)
	"""

	def calculate_PCF_phase(solutions, pi, coords, sx, sy, r_max, eu_side):
		# returns a numpy array: pcf = [[r, g], ...]; where r = radius of the annulus used to compute g(r), g = average correlation function g(r) based on all solutions

		# Number of particles in ring/area of ring/number of reference particles/number density
		# area of ring = pi*(r_outer**2 - r_inner**2)

		edges = np.arange(0., r_max + 1.1 * eu_side, eu_side)
		num_increments = len(edges) - 1
		radii = np.zeros(num_increments)

		for i in range(num_increments):
			radii[i] = (edges[i] + edges[i + 1]) / 2.
			r_outer = edges[i + 1]
			r_inner = edges[i]

		g = {}  # {si: g, ...}

		for si in range(solutions.shape[0]):

			coords_si = coords[solutions[si, :, pi]]
			coords_si -= coords_si.min(axis=0)
			x, y = coords_si.T

			# Find particles which are close enough to the box center that a circle of radius
			# r_max will not cross any edge of the box
			bools1 = x > r_max
			bools2 = x < (sx - r_max)
			bools3 = y > r_max
			bools4 = y < (sy - r_max)
			interior_indices, = np.where(bools1 * bools2 * bools3 * bools4)
			num_interior_particles = len(interior_indices)

			if num_interior_particles < 1:
				g[si] = None
				continue

			g[si] = np.zeros([num_interior_particles, num_increments])
			number_density = len(x) / (sx * sy)

			# Compute pairwise correlation for each interior particle
			for p in range(num_interior_particles):
				index = interior_indices[p]
				d = np.sqrt((x[index] - x) ** 2 + (y[index] - y) ** 2)
				d[index] = 2 * r_max
				d[np.isnan(d)] = 0

				(result, bins) = np.histogram(d, bins=edges, normed=False)
				g[si][p, :] = result / number_density

		# Average g(r) for all interior particles and compute radii
		g_average = np.zeros(num_increments)
		for i in range(num_increments):
			g_i = np.array([])
			for si in g:
				if not g[si] is None:
					g_i = np.hstack((g_i, g[si][:, i]))
			g_i = g_i[~np.isnan(g_i)]
			if g_i.size:
				g_average[i] = np.mean(g_i) / (np.pi * (r_outer ** 2 - r_inner ** 2))
			else:
				g_average[i] = np.nan

		return np.vstack((radii, g_average)).T

	# find maximum search radius for PCF
	sx = coords[:, 0].max() - coords[:, 0].min()
	sy = coords[:, 1].max() - coords[:, 1].min()
	r_max = min(sx, sy) / 4

	pcf = []
	for pi in range(solutions.shape[2]):
		pcf.append(calculate_PCF_phase(solutions, pi, coords, sx, sy, r_max, eu_side))
	pcf = np.array(
		pcf)  # pcf[pi] = [r, g]; where pi = index of phase, r = radius of the annulus used to compute g(r), g = average correlation function g(r)

	return pcf


def calculate_PCF_randomized(solutions, path_coords_examined, extent, eu_side, randomize_n):
	# calculate PCF for a randomized set of solutions, generated based on the actual solutions
	# inputs:
	#	solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	#	path_coords_examined = path in string format to a CSV file containing all examined coordinates
	#	extent
	#	eu_side = evidence unit square side (m)
	#	randomize_n = number of randomized solutions to generate when calculating the PCF
	# returns a list: pcf_randomized
	#	pcf_randomized[pi] = [[radii, g_lower, g_upper], ...]; where pi = index of phase; radii = [r, ...] and g_lower, g_upper = [g, ...]; in order of radii
	#		g = correlation function g(r)
	#		r = radius of the annulus used to compute g(r)
	#		g_lower, g_upper = 5th and 95th percentiles of randomly generated values of g for phase pi

	solutions_n = solutions.shape[0]
	phases_n = solutions.shape[2]

	# load coordinates of all examined units (walked fields and excavation sites)
	coords_examined = load_examined_coords(path_coords_examined)  # [[X, Y], ...]

	# reduce resolution of coords_examined to eu_side
	coords_examined = get_unique_2d((np.round(coords_examined / eu_side) * eu_side).astype(int))

	# crop coords_examined to extent of analysis
	coords_examined = coords_examined[(coords_examined[:, 0] > extent[0]) & (coords_examined[:, 0] < extent[1]) & (
				coords_examined[:, 1] > extent[2]) & (coords_examined[:, 1] < extent[3])]

	# find maximum search radius for PCF
	dx = coords_examined[:, 0].max() - coords_examined[:, 0].min()
	dy = coords_examined[:, 1].max() - coords_examined[:, 1].min()
	r_max = min(dx, dy) / 4

	# generate randomized solutions
	pcf_randomized = []  # pcf_randomized[pi] = [[radii, g_lower, g_upper], ...]; where radii = [r, ...] and g_lower, g_upper = [g, ...]; in order of radii
	coords_n = dict([(pi, int(round(sum([solutions[si, :, pi].sum() for si in range(solutions_n)]) / solutions_n))) for pi in range(phases_n)])
	# coords_n = {pi: n, ...}
	for pi in range(phases_n):
		res = {}  # {radius: g_r, ...}
		for ri in range(randomize_n):
			pcf_rnd = calculate_PCF(
				coords_examined[np.random.choice(coords_examined.shape[0], coords_n[pi], replace=False)], r_max,
				eu_side)  # [[r, g], ...]; where r = radius of the annulus used to compute g(r), g = average correlation function g(r) based on all solutions
			if pcf_rnd is not None:
				for r, g in pcf_rnd:
					if r is not None:
						r = int(round(r))
						if r not in res:
							res[r] = []
						res[r].append(g)
		radii = np.unique(list(res.keys())).astype(int)
		g_lower = np.zeros(radii.shape[0])
		g_upper = np.zeros(radii.shape[0])
		for i, r in enumerate(radii):
			g_lower[i] = np.percentile(res[r], 5)
			g_upper[i] = np.percentile(res[r], 95)
		pcf_randomized.append([radii, g_lower, g_upper])

	return pcf_randomized
