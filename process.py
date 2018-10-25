from matplotlib import pyplot
from sa_lib import *
import multiprocessing as mp
import numpy as np


PROC_N = 8 # number of processors to use for multiprocessing

INTERVAL_THRESH = 200
EU_SIDE = 100 # Evidence unit side (m)
PRODUCTION_AREA = 25 # Production area (ha)
WATER_LIMIT = 10
ADD_PHASE_AFTER = 200 # number of attempts after which to add a phase if no solution has been found
TIME_STEP = 100
RANDOMIZE_N = 1000

DISTRIBUTION = "uniform" # prior distribution used to determine absolute dating of phases
						# possible values are: "uniform" / "trapezoid" / "sigmoid"
TRANSITION_INTERVAL = 50

FDEM = "data/raster/dem.tif"
FSLOPE = "data/raster/slope.tif"
FWATER = "data/raster/water.tif"

FCOORDS = "data/coords_examined.csv"

FEVIDENCE = "data/evidence_br_ha.csv"
#FEVIDENCE = "data/evidence.csv"

if __name__ == '__main__':
	
	pool = mp.Pool(processes = PROC_N)
	
	
##### LOAD INPUT DATA

	print()
	print("creating descriptive system")
	
	data = load_input_data(FEVIDENCE) # [[BP_from, BP_to, X, Y], ...]
	
	coords, intervals, phases_chrono, intervals_coords, neighbours, exclude, production_areas, extent, raster_shape = get_descriptive_system(data, INTERVAL_THRESH, EU_SIDE, PRODUCTION_AREA, WATER_LIMIT, FDEM, FSLOPE, FWATER)
	# coords = [[X, Y], ...]; unique coordinates
	# intervals = [[BP_from, BP_to], ...]; unique dating intervals
	# phases_chrono[pi] = [[i, ...], ...]; chronological phases (groups of intervals which can be contemporary)
	#								where pi = index of phase and i = index in intervals
	# intervals_coords[i] = [[BP_from, BP_to], ...]; dating intervals for each coordinate 
	#												where i = index in coords
	# neighbours[i1] = [i2, ...]; neighbouring coordinates for each coordinate
	#							where i1, i2 are indices in coords
	# exclude = [[i1, i2], ...]; pairs of coordinates where their production areas spatially exclude each other
	#							where i1, i2 are indices in coords 
	# production_areas[k] = [[i, j], ...]; where k = index in coords; i, j = indices in cost_surface raster
	# extent = [xmin, xmax, ymin, ymax];
	# raster_shape = (rows, columns)
	
	
##### FIND POSSIBLE SOLUTIONS FOR ASSIGNING EVIDENCE UNITS TO CHRONO-SPATIAL PHASES WHILE MODELING PRODUCTION AREAS AROUND HABITATION AREAS

	print()
	print("finding solutions for chrono-spatial phasing of evidence units")
	print("evidence units:", len(coords))
	print("initial phases:", len(phases_chrono))
	print()
	
#	solutions, phases_spatial = find_solutions(intervals, phases_chrono, intervals_coords, neighbours, exclude, ADD_PHASE_AFTER, PROC_N, pool)
	
	import json # DEBUG
	with open("tmp_sol_br_ha.json", "r") as f: # DEBUG
		solutions, phases_spatial = json.load(f) # DEBUG
		solutions = np.array(solutions)
	
	phases_n = len(phases_spatial)
	
	# solutions[si, i, pi] = True/False; where si = index of solution, i = index in coords and pi = index of phase
	# phases_spatial[pi] = [[i, ...], ...]; where pi = index of phase and i = index in intervals
	# phases_n = number of chrono-spatial phases
	
##### ASSIGN ABSOLUTE DATINGS TO PHASES
	
	print()
	print()
	print("MCMC modeling absolute chronology of phases")
	print("phases:", len(phases_spatial))
	print()
	
	phase_intervals, pis = get_phase_intervals(intervals, phases_spatial)
	# phase_intervals[qi] = [BP_from, BP_to]; where qi = index in pis
	# pis = [pi, ...]; where pi = index of phase; ordered by earliest interval first
	
	'''
	chains = get_chains(phase_intervals, DISTRIBUTION, TRANSITION_INTERVAL)
	
	# chains = [chain, ...]; where chain[qi] = t; where qi = index in pis and t = time in calendar years BP
	
	time_phase_dist, ts = get_time_phase_distribution(chains, pis, phase_intervals)
	
	# time_phase_dist[ti, pi] = n; where ti = index in ts, pi = index of phase and n = number of incidences where phase pi dates to time ti
	# ts = [t, ...]; where t = absolute dating in years BP
	
	phase_datings = get_phase_datings(phase_intervals)
	# phase_datings[qi] = [BP_from, BP_to]; where qi = index in pis
	
	# re-order phase_intervals and phase_datings by phase indices
	pis = pis.tolist()
	phase_intervals = np.array([phase_intervals[pis.index(qi)].tolist() for qi in range(len(pis))], dtype = int)
	phase_datings = np.array([phase_datings[pis.index(qi)].tolist() for qi in range(len(pis))], dtype = int)
	'''
	
	# DEBUG start
	with open("tmp_tdist_%s_br_ha.json" % (DISTRIBUTION), "r") as f:
		phase_intervals, phase_datings, ts, time_phase_dist = json.load(f)
	phase_intervals = np.array(phase_intervals, dtype = np.uint16)
	phase_datings = np.array(phase_datings)
	ts = np.array(ts, dtype = int)
	time_phase_dist = np.array(time_phase_dist, dtype = int)
	# DEBUG end
	
	# phase_intervals[pi] = [BP_from, BP_to]; where pi = index in pis
	# phase_datings[pi] = [BP_from, BP_to]; where pi = index in pis
	
	
##### CALCULATE AMOUNTS OF HABITATION AREAS PER PHASE AND YEAR

	print()
	print()
	print("Calculating amounts of habitation areas per phase and year")
	
	# get indices of phases sorted by mean values of time intervals
	pis_sorted = np.argsort(phase_datings.sum(axis = 1) / 2)[::-1] # [pi, ...]; where pi = index of phase
	
	num_habi = sum_habitation_phases(solutions, neighbours)
	num_habi_t = mean_habitation_time(num_habi, time_phase_dist)
	num_habi_avg = num_habi.mean()

	# num_habi[pi, si] = amount of habitation areas; where ti = index in ts and si = index of solution
	# num_habi_t[ti] = mean amount of habitation areas; where ti = index in ts
	# num_habi_avg = average number of habitation units per year or phase
	
	
##### CALCULATE AMOUNT OF EVIDENCE PER YEAR
	
	print()
	print("Calculating amounts of evidence per year")
	
	data = load_input_data(FEVIDENCE) # data = [[BP_from, BP_to, X, Y], ...]
	ts_evid, num_evidence = sum_evidence(data)
	
	# ts_evid = [t, ...]; where t = absolute dating in years BP
	# num_evidence[ti] = summed probability of evidence dating to time t; where ti = index in ts_evid


##### CALCULATE HABITATION STABILITY INDEX (HSI)
	
	print()
	print("Calculating Habitation Stability Index")
	
	hsi_mean, hsi_mean_map = calculate_HSI(solutions, coords, EU_SIDE, time_phase_dist)
	hsi_mean_map = hsi_mean_map.mean(axis = 0)
	
	# hsi_mean[ti] = mean HSI; where ti = index in ts
	# hsi_mean_map[i] = mean HSI; where i = index in coords


##### CALCULATE SPATIAL OVERLAPPING OF HABITATION AREA UNITS FROM SUBSEQUENT PHASES
	
	print()
	print("Calculating spatial overlapping of habitation area units")
	
	overlapping, t_bins = calculate_overlapping(solutions, coords, ts, EU_SIDE, time_phase_dist, TIME_STEP)
	
	# overlapping[ti1, ti2] = r; where ti1, ti2 = indices in t_bins
	# t_bins = [t, ...]; where t = absolute dating of the beginning of the bin in years BP


##### ANALYSE SPATIAL CLUSTERING OF HABITATION AREAS BY CALCULATING A PAIR CORRELATION FUNCTION (PCF) FOR EVERY SOLUTION

	print()
	print("Calculating Pair Correlation Function of habitation areas")
	
	pcf = calculate_PCF_solutions(solutions, coords, EU_SIDE)
	pcf_randomized = calculate_PCF_randomized(solutions, FCOORDS, extent, EU_SIDE, RANDOMIZE_N)
	
	# pcf[pi] = [r, g]; where pi = index of phase, r = radius of the annulus used to compute g(r), g = average correlation function g(r)
	# pcf_randomized[pi] = [[radii, g_lower, g_upper], ...]; where pi = index of phase; radii = [r, ...] and g_lower, g_upper = [g, ...]; in order of radii
	# g = correlation function g(r)
	# r = radius of the annulus used to compute g(r)
	# g_lower, g_upper = 5th and 95th percentiles of randomly generated values of g for phase pi
	

##### GENERATE RASTER PROBABILITY MAPS OF MODELLED PRODUCTION AREAS FOR EVERY PHASE
	
	print()
	print("Generating raster probability maps of modelled production areas")

	pa_grids = generate_production_area_maps(solutions, raster_shape, neighbours, production_areas)
	
	# pa_grids[pi, i, j] = p; where pi = index of phase; i, j = indices in 2D raster with cell size = EU_SIDE; p = probability of presence of production area



##### PLOT AMOUNT OF HABITATION AREAS VS. AMOUNT OF EVIDENCE PER YEAR

	print()
	print("Plotting amount of habitation areas vs. amount of evidence per year")
	
	fig = pyplot.figure(figsize = (8, 6))
	
	tmin = min(ts.min(), ts_evid.min())
	tmax = max(ts.max(), ts_evid.max())
	num_evidence_avg = num_evidence.sum() / ts_evid.shape[0]

	pyplot.subplot(211)
	pyplot.plot(ts_evid, num_evidence)
	pyplot.plot([tmax, tmin], [num_evidence_avg, num_evidence_avg], color = "green")
	pyplot.xticks(ts_evid[::200], ts_evid[::200] - 1950)
	pyplot.xlim(tmax, tmin)

	pyplot.subplot(212)
	pyplot.plot(ts, num_habi_t, color = "k")
	#pyplot.plot(ts, running_mean(num_habi_t, ts, 100), color = "red")
	pyplot.plot([tmax, tmin], [num_habi_avg, num_habi_avg], color = "green")
	pyplot.xticks(ts[::200], ts[::200] - 1950)
	pyplot.xlim(tmax, tmin)

	pyplot.tight_layout()
	pyplot.savefig("_tmp/graph_01_evidence_vs_habitation.png")
	fig.clf()
	pyplot.close()


##### PLOT AVERAGE HSI PER YEAR

	print()
	print("Plotting average hsi per year")
	
	fig = pyplot.figure(figsize = (8, 3))
	pyplot.title("Mean Habitation Stability Index (HSI)")
	pyplot.plot(ts, hsi_mean)
	pyplot.xticks(ts[::200], ts[::200] - 1950)
	pyplot.xlim(ts.min(), ts.max())
	pyplot.tight_layout()
	pyplot.savefig("_tmp/graph_02_hsi.png")
	fig.clf()
	pyplot.close()


##### PLOT RASTER MAP OF AVERAGE HSI

	print()
	print("Plotting raster map of average hsi")
	
	water = GeoTIFF("data/raster/water.tif")
	water_cellsize = water.getCellSize()[0]
	water = water.getRaster() # water[x, y] = water flow
	water[water < 0] = 0
	water = (water * (water_cellsize**2)) / (100**2) # convert to hectares
	water = (water >= WATER_LIMIT)
	
	fig = pyplot.figure(figsize = (10, 8))
	pyplot.title("Mean Habitation Stability Index (HSI)")
	pyplot.imshow(water, extent = extent, cmap = "Blues")
	pyplot.scatter(coords[:,0], coords[:,1], c = hsi_mean_map, cmap = "Reds", s = 20, marker = "s")
	pyplot.colorbar()
	pyplot.xticks(rotation = 45)
	pyplot.tight_layout()
	pyplot.savefig("_tmp/graph_03_hsi_map.png")
	fig.clf()
	pyplot.close()


##### PLOT MATRIX OF SPATIAL OVERLAPPING OF HABITATION AREA UNITS FROM SUBSEQUENT PHASES

	print()
	print("Plotting matrix of spatial overlapping of habitation area units from subsequent phases")
	
	fig = pyplot.figure(figsize = (10, 8))
	pyplot.title("Spatial continuity of habitation areas")
	pyplot.matshow(overlapping, cmap = "jet")
	pyplot.colorbar()
	pyplot.tight_layout()
	pyplot.savefig("_tmp/graph_04_spatial_continuity.png")
	fig.clf()
	pyplot.close()


##### PLOT PCF FOR EVERY PHASE

	print()
	print("Plotting pcf for every phase")
	
	gmax = pcf[:,:,1].max()
	for _, _, g_upper in pcf_randomized:
		gmax = max(gmax, g_upper.max())
	
	fig = pyplot.figure(figsize = (8, 15))
	
	for pi in pis_sorted:
		pyplot.subplot(pis_sorted.shape[0], 1, pi + 1)
		radii, g_lower, g_upper = pcf_randomized[pi]
		pyplot.fill_between(radii, g_lower, g_upper, color = "lightgray")
		pyplot.plot(pcf[pi,:,0], pcf[pi,:,1], color = "k")
		pyplot.ylim(0, gmax)
	
	pyplot.tight_layout()
	pyplot.savefig("_tmp/graph_05_PCF.png")
	fig.clf()
	pyplot.close()
	

##### PLOT RASTER MAPS OF HABITATION AREAS FOR EVERY PHASE

	print()
	print("Plotting raster maps of habitation areas for every phase")
	
	ph = 1
	for pi in pis_sorted:
		
		print("\rplotting phase %d/%d    " % (ph, pis_sorted.shape[0]), end = "")
		
		ps = solutions[:,:,pi].mean(axis = 0)
		
		CE_from, CE_to = 1950 - phase_datings[pi]
		
		fig = pyplot.figure(figsize = (10, 8))
		pyplot.title("Phase %d (%d - %d CE)" % (ph, CE_from, CE_to))
		pyplot.imshow(water, extent = extent, cmap = "Blues")
		pyplot.scatter(coords[:,0], coords[:,1], c = ps, cmap = "Reds", s = 20, vmin = 0, vmax = 1, marker = "s")
		pyplot.colorbar()
		pyplot.xticks(rotation = 45)
		pyplot.tight_layout()
		pyplot.savefig("_tmp/map_habitation_ph_%03d.png" % (ph))
		fig.clf()
		pyplot.close()
		
		ph += 1

##### PLOT RASTER MAPS OF PRODUCTION AREAS FOR EVERY PHASE

	print()
	print()
	print("Plotting raster maps of production areas for every phase")
	
	ph = 1
	for pi in pis_sorted:
		
		print("\rplotting phase %d/%d    " % (ph, pis_sorted.shape[0]), end = "")
		
		grid = np.zeros(raster_shape)
		for solution in solutions:
			idxs = np.where(solution[:,pi])[0]
			clusters = find_clusters(idxs, neighbours)
			for cluster in clusters:
				collect = []
				for i in cluster:
					collect += production_areas[i].tolist()
				collect = np.array(collect, dtype = int)
				collect = np.unique(np.ascontiguousarray(collect).view(np.dtype((np.void, collect.dtype.itemsize * 2)))).view(collect.dtype).reshape(-1, 2)
				grid[collect[:,0], collect[:,1]] += 1
		grid /= solutions.shape[0]
		
		grid[grid == 0] = np.nan
		
		CE_from, CE_to = 1950 - phase_datings[pi]
		
		fig = pyplot.figure(figsize = (10, 8))
		pyplot.title("Phase %d (%d - %d CE)" % (ph, CE_from, CE_to))
		pyplot.imshow(water, extent = extent, cmap = "Blues")
		pyplot.imshow(grid, cmap = "Reds", extent = extent, vmin = 0, vmax = 1)
		pyplot.colorbar()
		pyplot.xticks(rotation = 45)
		pyplot.tight_layout()
		pyplot.savefig("_tmp/map_production_ph_%03d.png" % (ph))
		fig.clf()
		pyplot.close()
		
		ph += 1

