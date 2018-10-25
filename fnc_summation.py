import numpy as np

from fnc_chrono_spatial_modelling import (find_clusters)


def sum_habitation_phases(solutions, neighbours):
	# calculate temporal distribution of amount of modelled habitation areas, summed by modelled chrono-spatial phases
	# inputs:
	#	solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	#	neighbours[i1] = [i2, ...]; where i1, i2 are indices in coords
	# returns a list: num_habitation_areas[pi, si] = amount

	solutions_n = solutions.shape[0]
	phases_n = solutions.shape[2]

	# find clusters for all solutions and phases
	clusters = []  # clusters[si][pi] = [[i, ...], ...]
	for si in range(solutions_n):
		clusters.append([])
		for pi in range(phases_n):
			idxs = np.where(solutions[si, :, pi])[0]
			clusters[-1].append(find_clusters(idxs, neighbours))

	# sum amount of habitation areas per phase
	num_habitation_areas = np.zeros((phases_n, solutions_n))  # [pi, si] = amount
	for pi in range(phases_n):
		for si in range(solutions_n):
			num_habitation_areas[pi, si] += len(clusters[si][pi])

	return num_habitation_areas

def calc_mean_freq(frequencies, vals):
	# calculate mean from a frequency distribution
	# inputs:
	#	frequencies[ti, vi] = amount; vi = index in vals; ti = index in ts
	#	vals = [value, ...]
	# returns a numpy array: mean_freq[ti] = mean value; where ti = index in ts
	mean_freq = np.zeros(frequencies.shape[0])
	freq_sum = frequencies.sum(axis=1)
	mask = (freq_sum > 0)
	mean_freq[mask] = (frequencies * vals[None, :])[mask].sum(axis=1) / freq_sum[mask]  # mean_freq[ti] = mean value
	return mean_freq

def mean_habitation_time(num_habit, time_phase_dist):
	# calculate temporal distribution of amount of modelled habitation areas, summed by calendar years
	# inputs:
	#	num_habit[pi, si] = amount
	#	time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	# returns two numpy arrays: num_habitation_t, num_habitation_t_lower, num_habitation_t_upper
	# 	num_habitation_t[ti] = mean amount per year
	# 	num_habitation_t_lower[ti] = lower boundary of 90% interval
	# 	num_habitation_t_upper[ti] = upper boundary of 90% interval

	# get distribution of values in phases
	vals = np.unique(num_habit)
	vals.sort()
	ph_v_n = np.zeros((num_habit.shape[0], vals.shape[0]))  # [pi, vi] = n
	ph_s_v_n = np.zeros((num_habit.shape[1], num_habit.shape[0], vals.shape[0]))  # [si, pi, vi] = n
	for vi, val in enumerate(vals):
		ns = (num_habit == val).sum(axis=1)
		mask = (ns > 0)
		phs = np.where(mask)[0]
		ns = ns[mask]
		for i, pi in enumerate(phs):
			ph_v_n[pi, vi] += ns[i]
		for si in range(num_habit.shape[1]):
			phs = np.where(num_habit[:, si] == val)[0]
			for pi in phs:
				ph_s_v_n[si, pi, vi] = 1

	# calculate means
	num_habitation_t = calc_mean_freq((time_phase_dist[:, :, None] * ph_v_n[None, :, :]).sum(axis=1), vals)  # num_habitation_t[ti] = mean val
	num_habitation_t_sol = np.zeros((num_habit.shape[1], time_phase_dist.shape[0]))  # num_habitation_t_sol[si, ti] = mean val
	for si in range(num_habit.shape[1]):
		num_habitation_t_sol[si] = calc_mean_freq((time_phase_dist[:,:,None] * ph_s_v_n[si,None,:,:]).sum(axis = 1), vals)
	
	num_habitation_t_lower = np.zeros(num_habitation_t.shape)
	num_habitation_t_upper = np.zeros(num_habitation_t.shape)
	for ti in range(time_phase_dist.shape[0]):
		num_habitation_t_lower[ti] = np.percentile(num_habitation_t_sol[:,ti], 5)
		num_habitation_t_upper[ti] = np.percentile(num_habitation_t_sol[:,ti], 95)
	
	return num_habitation_t, num_habitation_t_lower, num_habitation_t_upper

def sum_evidence(data):
	# calculate temporal distribution of evidence, summed by calendar years
	# inputs:
	#	data = [[BP_from, BP_to, X, Y], ...]; where BP_from, BP_to = dating interval in calendar years BP; X, Y = coordinates of evidence unit
	# returns two numpy arrays: ts_evid, num_evidence
	#	ts_evid = [t, ...]; where t = absolute dating in years BP
	#	num_evidence[ti] = amount; where ti = index in ts_evid
	data = data[:, :2]  # [[BP_from, BP_to], ...]
	ts_evid = np.arange(data.min(), data.max())
	interval_lengths = data[:, 0] - data[:, 1]
	num_evidence = []  # num_evidence[ti] = amount; where ti = index in ts_evid
	for t in ts_evid:
		mask = ((t <= data[:, 0]) & (t > data[:, 1]))
		num_evidence.append((mask.astype(float) / interval_lengths).sum())
	return ts_evid, np.array(num_evidence)
