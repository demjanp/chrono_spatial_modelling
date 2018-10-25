import numpy as np
from scipy.spatial.distance import cdist


def calculate_overlapping(solutions, coords, ts, eu_side, time_phase_dist, time_step):
	# calculate continuity of habitation areas expressed as a ratio of units of a phase overlapping units from the previous phase
	# inputs:
	#	solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	ts = [t, ...]; where t = absolute dating in years BP
	#	eu_side = evidence unit square side (m)
	#	time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	#	time_step = time step in calendar years to use for binning temporal distributions
	# returns numpy arrays: overlapping, t_bins
	#	overlapping[ti1, ti2] = r; where ti1, ti2 = indices in t_bins
	#	t_bins = [t, ...]; where t = absolute dating of the beginning of the bin in years BP

	coords_n = coords.shape[0]
	solutions_n = solutions.shape[0]
	phases_n = time_phase_dist.shape[1]

	# bin time_phase_dist by time_step years
	t_bins = np.arange(ts.min(), ts.max(), time_step)
	tpd_binned = np.zeros((t_bins.shape[0], time_phase_dist.shape[1]), dtype=float)  # [ti, pi] = n; ti = index in t_bins
	for pi in range(time_phase_dist.shape[1]):
		for ti, t in enumerate(t_bins):
			tpd_binned[ti, pi] = time_phase_dist[((ts >= t) & (ts < t + time_step)), pi].sum()
	# convert each phase to a probability distribution
	tpd_binned /= tpd_binned.sum(axis=0)

	occupation_time = np.zeros((solutions_n, coords.shape[0], t_bins.shape[0]), dtype=float)  # [si, i, ti] = p; where si = solution index, i = index in coords, ti = index in t_bins, p = summed probability of occupation
	for si in range(solutions_n):
		for i in range(coords_n):
			for pi in range(phases_n):
				if solutions[si, i, pi]:
					occupation_time[si, i] += tpd_binned[:, pi]

	overlapping = np.zeros((t_bins.shape[0], t_bins.shape[0]), dtype=float)
	for ti1 in range(t_bins.shape[0]):
		for ti2 in range(t_bins.shape[0]):
			overlapping[ti1, ti2] = np.nan
			if ti1 < ti2:
				for si in range(solutions_n):
					slice1 = occupation_time[si, :, ti1]
					mask1 = (slice1 > 0)
					slice2 = occupation_time[si, :, ti2]
					mask2 = (slice2 > 0)
					if mask1.any() and mask2.any():
						coords1 = coords[mask1]
						coords2 = coords[mask2]
						d = cdist(coords1, coords2).min(axis=1)
						mask3 = (d < 3 * eu_side)
						if mask3.any:
							overlapping[ti1, ti2] = slice1[mask1][mask3].sum() / slice1.sum()

	return overlapping, t_bins
