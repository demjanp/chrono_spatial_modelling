import numpy as np
from osgeo import gdal, gdalnumeric

def load_input_data(path):
	# load evidence from a CSV file
	# inputs:
	#	path = path in string format to a CSV file containing the evidence (input data)
	# returns a numpy array: [[BP_from, BP_to, X, Y], ...]
	
	data = []
	with open(path, "r") as f:
		contents = f.read().split("\n")[1:]
		for row in contents:
			row = row.split(",")
			if len(row) == 4:
				bp_from, bp_to, x, y = [int(value) for value in row]
				data.append([bp_from, bp_to, x, y])
	data = np.array(data, dtype = int)
	return data

def load_examined_coords(path):
	# load all examined coordinates from a CSV file
	# inputs:
	#	path = path to file in string format
	# returns a numpy array: [[X, Y], ...]
	
	coords_examined = []
	with open(path, "r") as f:
			contents = f.read().split("\n")[1:]
			for row in contents:
				row = row.split(",")
				if len(row) < 2:
					break
				coords_examined.append([int(value) for value in row])
	return np.array(coords_examined, dtype = int)

class GeoTIFF(object):
	# handle operations with a GeoTIFF file (https://www.gdal.org/frmt_gtiff.html)
	
	def __init__(self, fsource):
		
		source = gdal.Open(fsource)
		self.proj_wkt = source.GetProjectionRef()
		self._x_min, self._cell_size_x, _, self._y_min, _, self._cell_size_y = source.GetGeoTransform()
		self._raster = source.GetRasterBand(1).ReadAsArray()
		self._height, self._width = self._raster.shape
		source = None
	
	def __getitem__(self, key):
		# return value at key = [x,y]; x, y in geographic coordinates
		# x, y can be np arrays
		
		if isinstance(key[0], np.ndarray):
			i, j, idxs = self.getIndex(key[0], key[1], get_idxs = True)
			out = np.zeros(key[0].shape[0])
			out[:] = np.nan
			if idxs.shape[0]:
				out[idxs] = self._raster[i[idxs], j[idxs]]
				return out
		else:
			i, j = self.getIndex(key[0], key[1])
			if not i is None:
				return self._raster[i, j]
		return None
	
	def getRaster(self):
		# return gdalnumeric matrix: [height, width] = value
		
		return self._raster
	
	def getIndex(self, x, y, get_idxs = False):
		# return index (i, j) in raster based on geographic coordinates (x, y)
		# x, y can be np arrays

		i = (np.round((y - self._y_min - self._cell_size_y / 2) / self._cell_size_y)).astype(int)
		j = (np.round((x - self._x_min - self._cell_size_x / 2) / self._cell_size_x)).astype(int)
		if isinstance(i, np.ndarray):
			idxs = np.where((i > 0) & (i < self._height) & (j > 0) & (j < self._width))[0]
			if get_idxs:
				return i, j, idxs
			elif idxs.shape[0]:
				return i[idxs], j[idxs]
		elif (i >= 0) and (i < self._height) and (j >= 0) and (j < self._width):
				return i, j
		return None, None
	
	def getCellSize(self):
		
		return self._cell_size_x, self._cell_size_y
	
	def getExtent(self):
		# return x_min, y_min, x_max, y_max
		
		return self._x_min, self._y_min + self._cell_size_y * self._height, self._x_min + self._cell_size_x * self._width, self._y_min
	
