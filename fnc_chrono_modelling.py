import numpy as np

from fnc_common import (get_unique_2d)


def get_phase_intervals(intervals, phases_spatial):
	# assign time intervals to phases by finding the highest lower and the lowest upper boundary of all intervals within each phase
	# inputs:
	#	intervals = [[BP_from, BP_to], ...]
	#	phases_spatial[pi] = [[i, ...], ...]; chrono-spatial phases; where pi = index of phase and i = index in intervals
	# returns a list: phase_intervals[pi] = [BP_from, BP_to]

	intervals = np.array(intervals, dtype=np.uint16)
	phase_intervals = []  # phase_intervals[pi] = [BP_from, BP_to]
	for phase in phases_spatial:
		phase = intervals[phase]  # [[BP_from, BP_to], ...]
		phase_intervals.append([phase[:, 0].min(), phase[:, 1].max()])
	phase_intervals = np.array(phase_intervals, dtype=np.int16)
	pis_sorted = np.argsort(phase_intervals.sum(axis=1) / 2)[::-1]
	phase_intervals = phase_intervals[pis_sorted]
	pis_sorted = np.arange(pis_sorted.shape[0])[pis_sorted]
	return phase_intervals, pis_sorted


def get_chains(phase_intervals, distribution):
	# generate Markov Chains for chronometric modelling
	"""
		Assign absolute dates to the modelled chrono-spatial phases under the assumption of a phase-sequence model, as described by 
		Bronk Ramsey, C. (2009) Bayesian analysis of radiocarbon dates. Radiocarbon, 51(1): 337-360.
		The original likelihoods for each phase are the dating intervals represented as uniform or trapezoid probability distributions.
	"""

	# inputs:
	#	distribution = "uniform" / "trapezoid"
	# returns a list: chains = [chain, ...]; where chain[qi] = t; where qi = index in pis and t = time in calendar years BP

	def get_chain(phase_intervals, func_pick_date):

		def get_phase_limits(phase_intervals):
			# each phase must have the BP_from boundary higher and the BP_to boundary lower than the previous phase
			# i.e each phase must begin after the start of the previous (older) phase and end before the end of the subsequent (younger) phase
			# returns a list: phase_limits[qi] = [BP_from, BP_to]

			phase_limits = phase_intervals.astype(int)  # phase_limits[qi] = [BP_from_limit, BP_to_limit]
			BP_from_limit, _ = phase_limits[0]
			phase_limits[0, 0] = BP_from_limit
			for qi in range(1, len(phase_limits)):
				BP_from, _ = phase_limits[qi]
				BP_from_limit = min(BP_from, BP_from_limit - 1)
				phase_limits[qi, 0] = BP_from_limit
			_, BP_to_limit = phase_limits[-1]
			phase_limits[-1, 1] = BP_to_limit
			for qi in range(len(phase_limits) - 1, -1, -1):
				_, BP_to = phase_limits[qi]
				BP_to_limit = max(BP_to, BP_to_limit + 1)
				phase_limits[qi, 1] = BP_to_limit
			return phase_limits

		phase_limits = get_phase_limits(phase_intervals)
		phases = np.arange(len(phase_intervals))
		chain = np.zeros(len(phase_intervals)).astype(int) - 1
		while chain[0] == -1:
			np.random.shuffle(phases)
			for pi0 in phases:
				t0 = func_pick_date(phase_intervals[pi0], phase_limits[pi0])
				if t0 is None:
					chain[:] = -1
					phase_limits = get_phase_limits(phase_intervals)
					break
				chain[pi0] = t0
				phase_limits[pi0] = [t0, t0]
				phase_limits = get_phase_limits(phase_limits)
		return chain.astype(np.uint16)

	def pick_date_uniform(interval, limits):
		# pick a date from interval based on a uniform distribution and constrained by limits
		# interval, limits = [BP_from, BP_to]

		tmin, tmax = max(interval[1], limits[1]), min(interval[0], limits[0])
		if tmin > tmax:
			return None
		if tmin == tmax:
			return tmin

		return np.random.randint(tmin, tmax)

	def pick_date_trapezoid(interval, limits):
		# pick a date from interval based on a trapezoid distribution and constrained by limits
		"""
			Implemented based on Karlsberg A. J. (2006) Flexible Bayesian methods for archaeological dating (PhD thesis). Sheffield: University of Sheffield.
		"""

		tmin, tmax = max(interval[1], limits[1]), min(interval[0], limits[0])
		if tmin > tmax:
			return None
		if tmin == tmax:
			return tmin

		# create a trapezoid prior probability distribution
		t1, t2 = interval[1], interval[0]
		d1 = np.random.randint(1, t2 - t1)
		d2 = np.random.randint(1, t2 - t1)
		a = t1 - d1 / 2
		b = t1 + d1 / 2
		c = t2 - d2 / 2
		d = t2 + d2 / 2
		h = 2 / (d + c - a - b)
		ts = np.arange(a, d)
		ps = np.zeros(ts.shape[0])
		mask = ((ts >= a) & (ts < b))
		ps[mask] = h * ((ts[mask] - a) / (b - a))
		ps[((ts >= b) & (ts < c))] = h
		mask = ((ts >= c) & (ts <= d))
		ps[mask] = h * ((d - ts[mask]) / (d - c))

		# crop the prior probability distribution to specified limits
		mask = ((ts >= tmin) & (ts <= tmax))
		ts = ts[mask]
		ps = ps[mask]
		ps /= ps.sum()

		# randomly pick from ts based on the prior probability distribution
		return np.random.choice(ts, 1, p=ps)[0]

	pick_date_fnc = {
		"uniform": pick_date_uniform,
		"trapezoid": pick_date_trapezoid,
	}

	chains = np.array([get_chain(phase_intervals, pick_date_fnc[distribution])], dtype=np.uint16)
	chains_mean0 = chains.mean(axis=0)
	counter = 0
	while counter < 1000:
		chains = np.vstack((chains, get_chain(phase_intervals, pick_date_fnc[distribution])))
		counter += 1
		chains_mean = chains.mean(axis=0)
		diff = (np.abs(chains_mean0 - chains_mean) / chains_mean0).max()
		if diff > 0.000001:
			counter = 0
		chains_mean0 = chains_mean
		if chains.shape[0] % 100 == 0:
			print("\rchains: %d  diff: %0.7f       " % (chains.shape[0], diff), end="")

	return chains


def get_time_phase_distribution(chains, pis, phase_intervals):
	# converts Markov chains into distributions of absolute dating of phases
	# inputs:
	#	chains = [chain, ...]; where chain[qi] = t; where qi = index in pis and t = time in calendar years BP
	#	pis = [pi, ...]; where pi = index of phase; ordered by earliest interval first
	#	phase_intervals[qi] = [BP_from, BP_to]; where qi = index in pis
	# returns two numpy arrays: time_phase_dist, ts
	#	time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	#	ts = [t, ...]; where t = absolute dating in years BP

	ts = np.arange(chains.min(), chains.max() + 1)  # [t, ...]; where t = absolute dating in years BP
	time_phase_dist = np.zeros((ts.shape[0], pis.shape[0]), dtype=int)  # [ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	for qi in range(len(phase_intervals)):
		freq = np.array([(chains[:, qi] == t).sum() for t in ts])
		time_phase_dist[:, pis[qi]] = freq
	return time_phase_dist, ts


def get_phase_datings(phase_intervals):
	# calculate absolute dating ranges for chrono-spatial phases (used of sorting of the phases)
	# inputs:
	#	phase_intervals[qi] = [BP_from, BP_to]; where qi = index in pis
	# returns a numpy array: phase_datings[qi] = [BP_from, BP_to]; where qi = index in pis 

	phase_intervals_unique = get_unique_2d(phase_intervals)
	phase_intervals_unique = phase_intervals_unique[np.argsort(phase_intervals_unique.mean(axis=1))]
	phase_datings = np.zeros((len(phase_intervals), 2), dtype=float)  # phase_datings[qi] = [BP_from, BP_to]
	for interval in phase_intervals_unique:
		qis = np.where((phase_intervals == interval).all(axis=1))[0]
		phase_len = (interval[0] - interval[1]) / qis.shape[0]
		BP_from = interval[0]
		BP_to = BP_from - phase_len
		for qi in qis:
			phase_datings[qi] = [BP_from, BP_to]
			BP_from = BP_to
			BP_to = BP_from - phase_len
	phase_datings = np.round(phase_datings).astype(np.int16)
	for interval in phase_intervals_unique:
		qis = np.where((phase_intervals == interval).all(axis=1))[0]
		if qis.shape[0] > 1:
			phase_datings[qis[:-1], 1] -= 1

	return phase_datings
