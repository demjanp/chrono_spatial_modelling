import numpy as np
import itertools
from collections import defaultdict

def find_clusters(idxs, neighbours_phase):
	# find clusters of neighbouring coords in idxs (idxs = indices in coords)
	# returns a list: [[i, ...], ...]; where i = index in coords
	
	if isinstance(neighbours_phase, list):
		neighbours_phase = dict([(i, neighbours_phase[i]) for i in range(len(neighbours_phase))])
	
	idxs = np.intersect1d(idxs, list(neighbours_phase.keys()))
	if not idxs.size:
		return []
	
	clusters = []
	done = []
	for i in idxs:
		if not i in done:
			todo = [i]
			cluster = []
			while todo:
				i1 = todo.pop()
				for i2 in np.intersect1d(neighbours_phase[i1], idxs):
					if not i2 in done:
						todo.append(i2)
						done.append(i2)
						cluster.append(i2)
			if cluster:
				clusters.append(cluster)
	return clusters

def find_solution(params):
	# returns a list: solution[i, pi] = True/False; where i = index in coords and pi = index of phase
	
	phases_n, coords_phases, neighbours_phases, exclude_phases, max_attempts = params
	
	coords_n = len(coords_phases)
	
	coords_phases = [[np.array(pis, dtype = np.uint16) for pis in coords_phases[i]] for i in range(coords_n)]
	exclude_phases = [np.array(rows, dtype = int) for rows in exclude_phases]
	
	solution = np.zeros((coords_n, phases_n), dtype = bool) # solution[i, pi] = True/False; where i = index in coords and pi = index of phase
	
	unassigned = dict([(i, [phases.copy() for phases in coords_phases[i]]) for i in range(coords_n)])
	# unassigned[i] = [phases, ...]; where phases = [pi, ...]
	counter = 0
	while unassigned and (counter < max_attempts):
		counter += 1
		
		# randomly assign unassigned evidence units to phases
		for i in unassigned:
			for phases in unassigned[i]:
				pi = np.random.choice(phases, 1)[0]
				solution[i, pi] = True
		
		# check in each phase if some units exclude each other
		found_exclusion = True
		solution_changed = False
		while found_exclusion:
			found_exclusion = False
			for pi in range(len(exclude_phases)):
				idxs = np.where(solution[:,pi])[0]
				if not idxs.size:
					# no units assigned to phase
					continue
				if not exclude_phases[pi].size:
					continue
				exclude_phase = exclude_phases[pi][np.in1d(exclude_phases[pi], idxs).reshape((-1,2)).all(axis = 1)]
				if not exclude_phase.size:
					# no potential mutually excluding units in the phase
					continue
				
				# check for exclusion between units which are members of different clusters
				clusters = find_clusters(idxs, neighbours_phases[pi])
				exclusions = []
				for c1, c2 in itertools.combinations(range(len(clusters)), 2):
					for i1 in clusters[c1]:
						for i2 in clusters[c2]:
							if ((exclude_phase == i1).any(axis = 1) & (exclude_phase == i2).any(axis = 1)).any():
								exclusions += [i1, i2]
				
				if exclusions:
					# remove one of the units with the most exclusions by other units
					found_exclusion = True
					exclusions = np.array(exclusions, dtype = np.uint16)
					exclusions = np.array([[i, (exclusions == i).sum()] for i in np.unique(exclusions)], dtype = np.uint16)
					exclusions, p = exclusions[:,0], exclusions[:,1].astype(float)
					p = p / p.sum()
					i = np.random.choice(exclusions, 1, p = p)[0]
					solution[i, pi] = False
					solution_changed = True
		
		unassigned = {}
		if solution_changed:
			for i in range(coords_n):
				unassigned[i] = []
				pis = np.where(solution[i])[0]
				found = False
				for phases in coords_phases[i]:
					if not np.intersect1d(pis, phases).size:
						unassigned[i].append(phases)
						found = True
				if not found:
					del unassigned[i]

	return solution, len(unassigned)			

def find_solutions(intervals, phases_chrono, intervals_coords, neighbours, exclude, add_phase_after, proc_n, pool):
	# returns two lists: solutions and chrono-spatial phases
	# solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	# phases[pi] = [[i, ...], ...]; where pi = index of phase and i = index in intervals
	
	def get_coords_phases(coords_idxs, intervals_coords, intervals, phases):
		# for each coord, find which phases it can belong to
		# returns two lists: coords_phases and coords_phases_all
		# coords_phases[i] = [phases, ...]; where i = index in coords and phases = [pi, ...]; pi = index of phase
		#									one phase of each list of phases per coordinate has to be assigned to the evidence unit
		# coords_phases_all[i] = [pi, ...]
		
		coords_phases = []
		coords_phases_all = []
		for i in coords_idxs:
			coords_phases.append([])
			coords_phases_all.append([])
			for from_to in intervals_coords[i]:
				coords_phase = []
				int = intervals.index(from_to)
				for pi in range(len(phases)):
					if int in phases[pi]:
						if pi not in coords_phase:
							coords_phase.append(pi)
						if pi not in coords_phases_all[-1]:
							coords_phases_all[-1].append(pi)
				coords_phases[-1].append(np.array(coords_phase, dtype = np.uint16))
			coords_phases_all[-1] = np.array(coords_phases_all[-1], dtype = np.uint16)
		return coords_phases, coords_phases_all
	
	def get_exclude_phases(phases, exclude, coords_phases_all):
		# returns a list: exclude_phases[pi] = [[i1, i2], ...]; where pi = index of phase and i1, i2 are indices in coords
		
		exclude_phases = []
		# exclude_phases[pi] = [[i1, i2], ...]
		for pi in range(len(phases)):
			exclude_phases.append([])
			for i1, i2 in exclude:
				if (coords_phases_all[i1] == pi).any() and (coords_phases_all[i2] == pi).any():
					exclude_phases[-1].append([i1, i2])
		exclude_phases = [np.array(row, dtype = np.uint16) for row in exclude_phases]
		return exclude_phases
	
	def get_neighbours_phases(phases, neighbours, coords_phases_all):
		# returns a dictionary: neighbours_phases = {pi: {i1: [i2, ...], ...}, ...}; where pi = index of phase and i1, i2 are indices in coords

		neighbours_phases = defaultdict(lambda: defaultdict(list))
		# neighbours_phases = {pi: {i1: [i2, ...], ...}, ...}
		for pi in range(len(phases)):
			for i1 in range(len(neighbours)):
				if (coords_phases_all[i1] == pi).any():
					for i2 in neighbours[i1]:
						if (coords_phases_all[i2] == pi).any():
							neighbours_phases[pi][i1].append(i2)
		return dict(neighbours_phases)
	
	def check_solutions_convergence(solutions, intervals_coords, intervals, phases):
		# returns mean phase_ratio
		# where phase_ratio = (number of unique phases assigned to evidence unit at a coordinate in all solutions) / (number of phases possible to assign to evidence unit at a coordinate)
		
		if not solutions:
			return 0
		
		_, coords_phases_all = get_coords_phases(range(len(intervals_coords)), intervals_coords, intervals, phases)
		# coords_phases_all[i] = [[pi, ...], ...]
		
		solutions = np.array(solutions, dtype = bool) # solutions[si,i,pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
		phase_ratios = np.array([len(np.unique(np.argwhere(solutions[:,i])[:,1])) / len(coords_phases_all[i]) for i in range(solutions.shape[1])], dtype = float)
		# phase_ratio = (number of unique phases assigned to evidence unit at a coordinate in all solutions) / (number of phases possible to assign to evidence unit at a coordinate)
		
		return phase_ratios.mean()
	
	phases = [phase.copy() for phase in phases_chrono]
	coords_n = len(intervals_coords)
	
	fitness = 0 # ratio of evidence units with an assigned phase
	best = 0
	phase_ratio = 0 # phase_ratio = (number of unique phases assigned to evidence unit at a coordinate in all solutions) / (number of phases possible to assign to evidence unit at a coordinate)
	phase_ratio_best = 0
	convergence_solutions0 = 0
	
	# find coordinates of evidence units, excludes and neighbours relevant for each phase
	coords_phases, coords_phases_all = get_coords_phases(range(coords_n), intervals_coords, intervals, phases) # coords_phases[i] = [[pi, ...], ...]; coords_phases_all[i] = [pi, ...]
	exclude_phases = get_exclude_phases(phases, exclude, coords_phases_all) # exclude_phases[pi] = [[i1, i2], ...]
	neighbours_phases = get_neighbours_phases(phases, neighbours, coords_phases_all) # neighbours_phases = {pi: {i1: [i2, ...], ...}, ...}
	
	solutions = []
	solutions_pool = []
	
	while True:
		
		print("\rMCMC fitness: %0.3f converg: %0.3f phases: %d solutions: %d     " % (best, phase_ratio, len(phases), len(solutions)), end = "")
		
		if not solutions_pool:
			params = proc_n * [[len(phases), [[pis.tolist() for pis in coords_phases[i]] for i in range(coords_n)], neighbours_phases, [rows.tolist() for rows in exclude_phases], add_phase_after if (best < 1) else add_phase_after * 2]]
			solutions_pool = pool.map(find_solution, params)
		
		solution, unassigned_n = solutions_pool.pop()
		
		fitness = (solution.shape[0] - unassigned_n) / solution.shape[0] # ratio of evidence units with an assigned phase
		
		if fitness > best:
			# if fitness has increased, reset solutions
			best = fitness
			solutions = []
		
		if fitness == best:
			# found a solution with current best fitness
			solution = solution.tolist()
			if solution not in solutions:
				solutions.append(solution)
		
		if (best < 1) and solutions and (not solutions_pool):
			# if fitness is not optimal and a solution has been found and the solution pool is empty
			# add a phase with the same dating range as one of the most unassigned phases
			
			# collect all unassigned phases
			unassigned_phases = []
			for i in range(coords_n):
				pis1 = np.where(solutions[0][i])[0]
				for pis2 in coords_phases[i]:
					if not np.intersect1d(pis1, pis2).size:
						unassigned_phases += pis2.tolist()
			
			# randomly choose one of the most occuring unassigned phases and duplicate it (add chrono-spatial phase)
			pis = np.array(unassigned_phases, dtype = int)
			pis = np.array([[pi, (pis == pi).sum()] for pi in np.unique(pis)], dtype = int)
			pis = pis[pis[:,1] == pis[:,1].max(),0]
			pi = np.random.choice(pis, 1)[0]
			phases.append(phases[pi])
			
			# re-calculate coordinates, excludes and neighbours relevant for each phase
			coords_phases, coords_phases_all = get_coords_phases(range(coords_n), intervals_coords, intervals, phases) # coords_phases[i] = [[pi, ...], ...]; coords_phases_all[i] = [pi, ...]
			exclude_phases = get_exclude_phases(phases, exclude, coords_phases_all) # exclude_phases[pi] = [[i1, i2], ...]
			neighbours_phases = get_neighbours_phases(phases, neighbours, coords_phases_all) # neighbours_phases = {pi: {i1: [i2, ...], ...}, ...}
		
		if best == 1:
			# if fitness is optimal, check solutions for convergence (if all units have been assigned all possible phases in the set of solutions)
			phase_ratio = check_solutions_convergence(solutions, intervals_coords, intervals, phases)
			if phase_ratio == 1:
				break
			if phase_ratio > phase_ratio_best:
				phase_ratio_best = phase_ratio
				convergence_solutions0 = len(solutions)
			# if number of solutions has increased by 10 since phase_ratio value has last increased, stop processing
			if len(solutions) >= convergence_solutions0 + 10:
				break
	
	return np.array(solutions, dtype = bool), phases

