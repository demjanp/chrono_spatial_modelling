import numpy as np

def calculate_HSI(solutions, coords, eu_side, time_phase_dist):
	# calculate Habitation Stability Index, which is the ratio of habitation area units which are close to units inhabited in the previous phase
	# inputs:
	#	solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	#	coords = [[X, Y], ...]; unique coordinates of evidence units
	#	eu_side = evidence unit square side (m)
	#	time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	# returns numpy arrays: hsi_mean, hsi_mean_lower, hsi_mean_upper, hsi_mean_map
	#	hsi_mean[ti] = mean HSI; where ti = index in ts
	#	hsi_mean_lower[ti] = lower boundary of 90% interval
	#	hsi_mean_upper[ti] = upper boundary of 90% interval
	#	hsi_mean_map[pi, i] = mean HSI; where pi = index of phase and i = index in coords
	
	ts_n = time_phase_dist.shape[0]
	phases_n = time_phase_dist.shape[1]
	solutions_n = solutions.shape[0]

	# find neighbours within 3 * eu_side for each coordinate (cca 28 ha ~= close settlement area)
	neighbours_3 = [] # neighbours_3[i1] = [i2, ...]; where i1, i2 = indices in coords
	for i1 in range(coords.shape[0]):
		neighbours_3.append([i1])
		for i2 in range(coords.shape[0]):
			if (i1 != i2) and (abs(coords[i1,0] - coords[i2,0]) < 3 * eu_side) and (abs(coords[i1,1] - coords[i2,1]) < 3 * eu_side) and (((coords[i1] - coords[i2])**2).sum() < (3 * eu_side)**2):
				neighbours_3[-1].append(i2)

	# convert each phase to a probability distribution
	tpd = time_phase_dist / time_phase_dist.sum(axis = 0)[None,:] # [ti, ph] = p; where ti = index in ts, pi = index of phase and p = probability of phase dating to time ti

	# find habit. units & number of stable habit. units for each phase and solution
	presence = np.zeros((phases_n, coords.shape[0], solutions_n), dtype = float) # [pi, i, si] = 1/0
	count = np.zeros((phases_n, solutions_n, 2), dtype = float) # [pi, si] = [n_habitation, n_stable]
	for pi in range(phases_n - 1):
		for i1 in range(coords.shape[0]):
			for si in range(solutions_n):
				if solutions[si, i1, pi]:
					count[pi, si, 0] += 1
					for i2 in neighbours_3[i1]:
						if solutions[si, i2, pi + 1]:
							presence[pi, i1, si] = 1
							count[pi, si, 1] += 1
							break

	# calculate stability index (HSI) = no. of stable habit. units / no. of all habit. units
	count_time = np.zeros((ts_n, solutions_n, 2), dtype = float) # [ti, si] = [p_habi, p_stable]
	for pi in range(phases_n):
		count_time += (count[None,pi,:,:] * tpd[:,pi,None,None])
	
	hsi_mean_map = presence.mean(axis = 2) # [pi,i] = mean p
	
	hsi_mean = count_time.sum(axis = 1)
	hsi_mean = hsi_mean[:,1] / hsi_mean[:,0]
	hsi_mean[np.isnan(hsi_mean) | (hsi_mean == np.inf)] = 0
	
	hsi_mean_sol = (count_time[:,:,1] / count_time[:,:,0]).T # hsi_mean_sol[si, ti] = mean p
	hsi_mean_sol[np.isnan(hsi_mean_sol) | (hsi_mean_sol == np.inf)] = 0
	
	hsi_mean_lower = np.zeros(hsi_mean.shape) # hsi_mean_lower[ti] = lower boundary of 90% interval
	hsi_mean_upper = np.zeros(hsi_mean.shape) # hsi_mean_upper[ti] = upper boundary of 90% interval
	for ti in range(ts_n):
		hsi_mean_lower[ti] = np.percentile(hsi_mean_sol[:,ti], 5)
		hsi_mean_upper[ti] = np.percentile(hsi_mean_sol[:,ti], 95)
	
	return hsi_mean, hsi_mean_lower, hsi_mean_upper, hsi_mean_map

