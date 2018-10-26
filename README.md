## Example code for the paper "Modelling prehistoric settlement activities based on surface and subsurface surveys" by D. Dreslerová and P. Demján

## Author of the code:
Peter Demján (peter.demjan@gmail.com)

## Overview
This is an example implementation of chrono-spatial modelling of settlement activities used in the aforementioned paper. The purpose is to provide a functional overview of the used methods, which is possible to execute and run without access to supercomputing facilities. Sample data is provided, which represents a subset of the actual data used in the original study. As of 2018, the processing should take under 48 hours on a reasonably powerful personal computer. The algorithms are intentionally only weakly optimized to preserve better readability of the code.

The processing includes the following steps:
1. Load input data and generate a descriptive system
2. Generate a set of solutions of chrono-spatial phasing of the evidence using MCMC
3. Assign absolute dating to the modelled chrono-spatial phases in form of posterior probability distributions generated using MCMC analysis
4. Calculate temporal distribution of amount of modelled habitation areas
5. Calculate temporal distribution of summed evidence
6. Calculate temporal distribution of the Habitation Stability Index (HSI)
7. Analyze habitation continuity by calculating ratio of overlapping of habitation areas in subsequent phases
8. Analyze spatial clustering of habitation areas by calculating a Pair Correlation Function (PCF) for every solution
9. Generate raster probability maps of modelled production areas for every chrono-spatial phase
10. Plot the results

## How to run

The script is implemented in the [Python](https://www.python.org/) programming language. All libraries used are open-source and available online (for details see Dependencies)

To execute, run [process.py](process.py)

Processing parameters are set by editing the values at the beginning of the [process.py](process.py) file.

It is possible to set different prior distributions for the chronometric modelling (step 3): uniform and trapezoid. This simulates different transitions between archaeological cultures, which were used to obtain the absolute dating intervals of the evidence units. The trapezoid distribution is modelled according to: Karlsberg A.J. 2006. Flexible Bayesian methods for archaeological dating (PhD thesis). Sheffield: University of Sheffield.

The code to compute the Pair Correlation Function is created according to an [example](https://github.com/cfinch/Shocksolution_Examples/tree/master/PairCorrelation) by Craig Finch.

Input data are loaded from the directory [data](data) and have the form of CSV files and [GeoTIFF](https://www.gdal.org/frmt_gtiff.html) rasters.
* [evidence.csv](data/evidence.csv) - dating<sup>1</sup> and coordinates<sup>2</sup> of all units of settlement evidence from field walking and excavations used in the original study
* [evidence_example.csv](data/evidence_example.csv) - dating<sup>1</sup> and coordinates<sup>2</sup> of selected units of settlement evidence used in this example code
* [coords_examined.csv](data/coords_examined.csv) - coordinates<sup>2</sup> of all examined units, regardles of presence or dating of settlement evidence (used for randomization)
* [dem.tif](data/raster/dem.tif) - Digital Elevation Model of the section of ladscape used in this example code
* [slope.tif](data/raster/slope.tif) - slope values of the aforementioned section, calculated in [ArcGIS](http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/slope.htm)
* [water.tif](data/raster/water.tif) - flow accumulation of the aforementioned section, calculated in [ArcGIS](http://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/flow-accumulation.htm)

<sup>1</sup> all datings are in calendar years BP <br>
<sup>2</sup> all coordinates are in the Pulkovo 1942 / Gauss-Krüger zone 13 coordinate system and projection (EPSG: 28403)

Results are plotted in from of graphs and maps in the directory 'output'.
Pre-computed example results calculated with different prior distributions are available in the directory 'example_output':
* [uniform](example_output/uniform)
* [trapezoid](example_output/trapezoid)

## Requirements

Running the script requires [Python 3.6](https://www.python.org/)

## Dependencies

The script requires the following libraries to be installed:
* [NumPy](http://www.numpy.org/): pip install numpy
* [SciPy](https://www.scipy.org/): pip install scipy
* [GDAL](http://www.gdal.org/): pip install GDAL

## License:
This code is licensed under the [MIT License](http://opensource.org/licenses/MIT) - see the [LICENSE.md](LICENSE.md) file for details

