import numpy as np

from fnc_chrono_spatial_modelling import (find_clusters)

def sum_habitation_phases(solutions, neighbours):
	# returns a list: num_habitation_areas[pi, si] = amount
	
	solutions_n = solutions.shape[0]
	phases_n = solutions.shape[2]
	
	# find clusters for all solutions and phases
	clusters = [] # clusters[si][pi] = [[i, ...], ...]
	for si in range(solutions_n):
		clusters.append([])
		for pi in range(phases_n):
			idxs = np.where(solutions[si,:,pi])[0]
			clusters[-1].append(find_clusters(idxs, neighbours))

	# sum amount of habitation areas per phase
	num_habitation_areas = np.zeros((phases_n, solutions_n)) # [pi, si] = amount
	for pi in range(phases_n):
		for si in range(solutions_n):
			num_habitation_areas[pi, si] += len(clusters[si][pi])
	
	return num_habitation_areas

def mean_habitation_time(num_habit, time_phase_dist):
	# num_habit[pi, si] = amount
	# time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	# return num_habitation_t[ti] = mean amount
	
	# get distribution of values in phases
	vals = np.unique(num_habit)
	vals.sort()
	ph_v_n = np.zeros((num_habit.shape[0], vals.shape[0])) # [pi, vi] = n
	ph_s_v_n = np.zeros((num_habit.shape[1], num_habit.shape[0], vals.shape[0])) # [si, pi, vi] = n
	for vi, val in enumerate(vals):
		ns = (num_habit == val).sum(axis = 1)
		mask = (ns > 0)
		phs = np.where(mask)[0]
		ns = ns[mask]
		for i, pi in enumerate(phs):
			ph_v_n[pi, vi] += ns[i]
		for si in range(num_habit.shape[1]):
			phs = np.where(num_habit[:,si] == val)[0]
			for pi in phs:
				ph_s_v_n[si, pi, vi] = 1
	
	# calculate mean
	num_habitation_t = np.zeros(time_phase_dist.shape[0])
	data = (time_phase_dist[:,:,None] * ph_v_n[None,:,:]).sum(axis = 1) # [ti, vi] = amount; vi = index in vals; ti = index in ts
	data_sum = data.sum(axis = 1)
	mask = (data_sum > 0)
	num_habitation_t[mask] = (data * vals[None,:])[mask].sum(axis = 1) / data_sum[mask] # num_habitation_t[ti] = mean val
	
	return num_habitation_t

def sum_evidence(data):
	
	# sum amount of evidence per calendar year
	data = data[:,:2] # [[BP_from, BP_to], ...]
	ts_evid = np.arange(data.min(), data.max())
	interval_lengths = data[:,0] - data[:,1]
	num_evidence = [] # num_evidence[ti] = amount; where ti = index in ts_evid
	for t in ts_evid:
		mask = ((t <= data[:,0]) & (t > data[:,1]))
		num_evidence.append((mask.astype(float) / interval_lengths).sum())
	return ts_evid, np.array(num_evidence)

