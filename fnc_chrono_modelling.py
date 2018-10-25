import numpy as np

from fnc_common import (get_unique_2d)

def get_phase_intervals(intervals, phases_spatial):
	# assign time intervals to phases by finding the highest lower and the lowest upper boundary of all intervals within each phase
	# returns a list: phase_intervals[pi] = [BP_from, BP_to]
	
	intervals = np.array(intervals, dtype = np.uint16)
	phase_intervals = [] # phase_intervals[pi] = [BP_from, BP_to]
	for phase in phases_spatial:
		phase = intervals[phase] # [[BP_from, BP_to], ...]
		phase_intervals.append([phase[:,0].min(), phase[:,1].max()])
	phase_intervals = np.array(phase_intervals, dtype = np.int16)
	pis_sorted = np.argsort(phase_intervals.sum(axis = 1) / 2)[::-1]
	phase_intervals = phase_intervals[pis_sorted]
	pis_sorted = np.arange(pis_sorted.shape[0])[pis_sorted]
	return phase_intervals, pis_sorted

def get_chains(phase_intervals, distribution, t_transition):
	# distribution = "uniform" / "trapezoid" / "sigmoid"
	# t_transition = length of the transition periods at the beginning and end of the distribution (only valid for trapezoid and sigmoid distributions)
	# returns a list: chains = [chain, ...]; where chain[qi] = t; where qi = index in pis and t = time in calendar years BP
	
	def get_dist_lookup(phase_intervals, t_transition):
		# generate lookup tables for trapezoid and sigmoid distributions (see Karlsberg 2006) to speed up generating dates
		
		dist_lookup_trapezoid = {}
		dist_lookup_sigmoid = {}
		for t2, t1 in phase_intervals:
			t_trans = min(t_transition, t2 - t1)
			a = t1 - t_trans/2
			b = t1 + t_trans/2
			c = t2 - t_trans/2
			d = t2 + t_trans/2
			h = 2 / (d + c - a - b)
			ts = np.arange(a, d)
			ps_trapezoid = np.zeros(ts.shape[0])
			ps_sigmoid = np.zeros(ts.shape[0])
			gs = np.linspace(0, 1, t_trans)
			gs = gs**2 / (gs**2 + (1 - gs)**2)
			
			mask = ((ts >= a) & (ts < b))
			ps_trapezoid[mask] = h * ((ts[mask] - a) / (b - a))
			ps_sigmoid[mask] = ps_trapezoid[mask] * gs
			
			mask = ((ts >= b) & (ts < c))
			ps_trapezoid[mask] = h
			ps_sigmoid[mask] = h
			
			mask = ((ts >= c) & (ts <= d))
			ps_trapezoid[mask] = h * ((d - ts[mask])/(d - c))
			ps_sigmoid[mask] = ps_trapezoid[mask] * (1 - gs)
			
			key = "%d_%d" % (t2, t1)
			dist_lookup_trapezoid[key] = [ts, ps_trapezoid]
			dist_lookup_sigmoid[key] = [ts, ps_sigmoid]
		return dist_lookup_trapezoid, dist_lookup_sigmoid
	
	def get_chain(phase_intervals, func_pick_date, dist_lookup):
		
		def get_phase_limits(phase_intervals):
			# each phase must have the BP_from boundary higher and the BP_to boundary lower than the previous phase
			# i.e each phase must begin after the start of the previous (older) phase and end before the end of the subsequent (younger) phase
			# returns a list: phase_limits[qi] = [BP_from, BP_to]
			
			phase_limits = phase_intervals.astype(int) # phase_limits[qi] = [BP_from_limit, BP_to_limit]
			BP_from_limit, _ = phase_limits[0]
			phase_limits[0,0] = BP_from_limit
			for qi in range(1, len(phase_limits)):
				BP_from, _ = phase_limits[qi]
				BP_from_limit = min(BP_from, BP_from_limit - 1)
				phase_limits[qi,0] = BP_from_limit
			_, BP_to_limit = phase_limits[-1]
			phase_limits[-1,1] = BP_to_limit
			for qi in range(len(phase_limits) - 1, -1, -1):
				_, BP_to = phase_limits[qi]
				BP_to_limit = max(BP_to, BP_to_limit + 1)
				phase_limits[qi,1] = BP_to_limit
			return phase_limits
		
		phase_limits = get_phase_limits(phase_intervals)
		phases = np.arange(len(phase_intervals))
		chain = np.zeros(len(phase_intervals)).astype(int) - 1
		while chain[0] == -1:
			np.random.shuffle(phases)
			for pi0 in phases:
				t0 = func_pick_date(phase_intervals[pi0], phase_limits[pi0], dist_lookup)
				if t0 is None:
					chain[:] = -1
					phase_limits = get_phase_limits(phase_intervals)
					break
				chain[pi0] = t0
				phase_limits[pi0] = [t0, t0]
				phase_limits = get_phase_limits(phase_limits)
		return chain.astype(np.uint16)

	def pick_date_uniform(interval, limits, lookup):
		# pick a date from interval based on a uniform distribution and constrained by limits
		# interval, limits = [BP_from, BP_to]
		
		tmin, tmax = max(interval[1], limits[1]), min(interval[0], limits[0])
		if tmin > tmax:
			return None
		if tmin == tmax:
			return tmin
		
		return np.random.randint(tmin, tmax)

	def pick_date_nonuniform(interval, limits, lookup):
		# pick a date from interval based on a non-uniform distribution and constrained by limits
		
		tmin, tmax = max(interval[1], limits[1]), min(interval[0], limits[0])
		if tmin > tmax:
			return None
		if tmin == tmax:
			return tmin
		
		ts, ps = lookup["%d_%d" % tuple(interval.tolist())]
		
		mask = ((ts >= tmin) & (ts <= tmax))
		ts = ts[mask]
		ps = ps[mask]
		ps /= ps.sum()
		
		return np.random.choice(ts, 1, p = ps)[0]
	
	dist_lookup_trapezoid, dist_lookup_sigmoid = get_dist_lookup(phase_intervals, t_transition)
	
	pick_date_fnc = {
		"uniform": [pick_date_uniform, None],
		"trapezoid": [pick_date_nonuniform, dist_lookup_trapezoid],
		"sigmoid": [pick_date_nonuniform, dist_lookup_sigmoid],
	}
	
	chains = np.array([get_chain(phase_intervals, *pick_date_fnc[distribution])], dtype = np.uint16)
	chains_mean0 = chains.mean(axis = 0)
	counter = 0
	while (counter < 1000):
		chains = np.vstack((chains, get_chain(phase_intervals, *pick_date_fnc[distribution])))
		counter += 1
		chains_mean = chains.mean(axis = 0)
		diff = (np.abs(chains_mean0 - chains_mean) / chains_mean0).max()
		if diff > 0.000001:
			counter = 0
		chains_mean0 = chains_mean
		if chains.shape[0] % 100 == 0:
			print("\rchains: %d  diff: %0.6f       " % (chains.shape[0], diff), end = "")
	
	return chains

def get_time_phase_distribution(chains, pis, phase_intervals):
	# converts Markov chains into distributions of absolute dating of phases
	# returns two numpy arrays: time_phase_dist, ts
	# time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	# ts = [t, ...]; where t = absolute dating in years BP
	
	ts = np.arange(chains.min(), chains.max() + 1) # [t, ...]; where t = absolute dating in years BP
	time_phase_dist = np.zeros((ts.shape[0], pis.shape[0]), dtype = int) # [ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	for qi in range(len(phase_intervals)):
		freq = np.array([(chains[:,qi] == t).sum() for t in ts])
		time_phase_dist[:,pis[qi]] = freq
	return time_phase_dist, ts

def get_phase_datings(phase_intervals):
	# calculate absolute dating ranges for phases
	# returns a numpy array: phase_datings[qi] = [BP_from, BP_to]; where qi = index in pis 
	
	phase_intervals_unique = get_unique_2d(phase_intervals)
	phase_intervals_unique = phase_intervals_unique[np.argsort(phase_intervals_unique.mean(axis = 1))]
	phase_datings = np.zeros((len(phase_intervals), 2), dtype = float) # phase_datings[qi] = [BP_from, BP_to]
	for interval in phase_intervals_unique:
		qis = np.where((phase_intervals == interval).all(axis = 1))[0]
		phase_len = (interval[0] - interval[1]) / qis.shape[0]
		BP_from = interval[0]
		BP_to = BP_from - phase_len
		for qi in qis:
			phase_datings[qi] = [BP_from, BP_to]
			BP_from = BP_to
			BP_to = BP_from - phase_len
	phase_datings = np.round(phase_datings).astype(np.int16)
	for interval in phase_intervals_unique:
		qis = np.where((phase_intervals == interval).all(axis = 1))[0]
		if qis.shape[0] > 1:
			phase_datings[qis[:-1],1] -= 1
	
	return phase_datings

