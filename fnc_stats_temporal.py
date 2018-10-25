import numpy as np

def get_dating_intervals(data):
	# returns a list of unique dating intervals: [[BP_from, BP_to], ...]
	
	intervals = []
	for BP_from, BP_to, _, _ in data:
		if [BP_from, BP_to] not in intervals:
			intervals.append([int(BP_from), int(BP_to)])
	return intervals

def get_chronological_phases(intervals, interval_thresh):
	# get chronological phases (groups of intervals which can be contemporary)
	# returns two lists: phases, contemporary
	# phases[pi] = [[i, ...], ...]; where pi = index of phase and i = index in intervals
	# contemporary[i1] = [i2, ...]; where i1, i2 are indices in intervals
	
	# for each interval, find all other intervals that can be contemporary
	contemporary = []
	for i1 in range(len(intervals)):
		contemporary.append([])
		for i2 in range(len(intervals)):
			if i1 != i2:
				min_rng = min(intervals[i1][0] - intervals[i1][1], intervals[i2][0] - intervals[i2][1])
				ovr_rng = min(intervals[i1][0], intervals[i2][0]) - max(intervals[i1][1], intervals[i2][1])
				if ((min_rng >= interval_thresh) and (ovr_rng == min_rng)) or ((min_rng < interval_thresh) and (ovr_rng > 1)):
					# i1 and i2 can be contemporary (they overlap by the whole shorter interval or they overlap and the shorther interval is < interval_thresh)
					contemporary[-1].append(i2)
	
	# find phases (groups of intervals where each interval must be possibly contemporary with each other interval in its group)
	phases = [] # [[i, ...], ...]; i = index in intervals
	added = True
	while added:
		added = False
		for i1 in range(len(intervals)):
			found = False
			for pi, phase in enumerate(phases):
				if not i1 in phase:
					found = True
					for i2 in phase:
						if not (i2 in contemporary[i1]):
							found = False
					if found:
						phases[pi].append(i1)
						added = True
			if not found:
				for phase in phases:
					if i1 in phase:
						found = True
						break
				if not found:
					phases.append([i1])
					added = True
			
	# find highest lower and lowest upper time boundary for each phase
	min_ranges_phases = {} # {pi: [bp_to, bp_from], ...}; pi = index in phases
	for pi, phase in enumerate(phases):
		min_ranges_phases[pi] = [-np.inf, np.inf]
		for i in phase:
			min_ranges_phases[pi] = [max(min_ranges_phases[pi][0], intervals[i][1]), min(min_ranges_phases[pi][1], intervals[i][0])]

	# sort by mean values of time intervals of phases
	for pi in range(len(phases)):
		phases[pi] = sorted(phases[pi], key = lambda from_to: np.mean(from_to))
	
	return phases, contemporary

def get_intervals_per_coords(data, coords):
	# get dating intervals for each coordinate
	# returns a list: intervals_coords[i] = [[BP_from, BP_to], ...]; where i = index in coords
	
	intervals_coords = []
	for X, Y in coords:
		# if one dating interval falls into another less precise interval, settlement is *required* to be present in the more precise interval
		intervals_coord = data[(data[:,2] == X) & (data[:,3] == Y), :2].astype(int) # [[BP_from, BP_to], ...]
		collect = [] # [[BP_to, BP_from], ..]
		for BP_from, BP_to in intervals_coord:
			if not [BP_to, BP_from] in collect:
				collect.append([BP_to, BP_from])
		collect_intervals = []
		for BP_to1, BP_from1 in collect:
			inside = False
			for BP_to2, BP_from2 in collect:
				if ([BP_to2, BP_from2] != [BP_to1, BP_from1]) and (BP_to2 >= BP_to1) and (BP_from2 <= BP_from1):
					inside = True
					break
			if not inside:
				collect_intervals.append([int(BP_from1), int(BP_to1)])
		intervals_coords.append(sorted(collect_intervals))
	return intervals_coords

