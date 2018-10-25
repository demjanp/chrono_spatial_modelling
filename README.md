## Example code for the paper "Modelling prehistoric settlement activities based on surface and subsurface surveys" by D. Dreslerová and P. Demján

## Citation:
TODO

## Author of the code:
Peter Demján (peter.demjan@gmail.com)

## Overview
This is an example implementation of chrono-spatial modelling of settlement activities used in the aforementioned paper. The purpose is to provide a functional overview of the used methods, which is possible to execute and run without access to supercomputing facitities. Sample data is provided, which represents a subset of the actual data used in the original study. As of 2018, the processing should take under 24 hours on a reasonably powerful personal computer.

The processing includes the following steps:
1. Load input data and generate a descriptive system
2. Generate a set of solutions of chrono-spatial phasing of the evidence using MCMC
3. Assign absolute dating to the modelled chrono-spatial phases in form of probability distributions generated using MCMC
4. Calculate temporal distribution of amount of modelled habitation areas
5. Calculate temporal distribution of summed evidence
6. Calculate temporal distribution of the Habitation Stability Index (HSI)
7. Analyse habitation continuity by calculating ratio of overlapping of habitation areas in subsequent phases
8. Analyse spatial clustering of habitation areas by calculating a Pair Correlation Function (pcf) for every solution
9. Generate raster probability maps of modelled production areas for every chrono-spatial phase
10. Plot the results

## How to run

The script is implemented in the [Python](https://www.python.org/) programming language. All libraries used are open-source and available online (for details see Dependencies)

To execute, run [process.py](process.py)

Processing parameters are set by editing the values at the beginning of the [process.py](process.py) file.

It is possible to set different prior distributions for the chronometric modelling (step 3): uniform, trapezoid and sigmoid. This simulates different transitions between archaeological cultures, which were used to obtain the absolute dating intervals of the evidence units. The duration of the transition interval can also be set. The non-uniform distributions are modelled according to: Karlsberg A.J. 2006. Flexible Bayesian methods for archaeological dating (PhD thesis). Sheffield: University of Sheffield.

Input data are loaded from the directory [data](data) and have the form of CSV files and [GeoTIFF](https://www.gdal.org/frmt_gtiff.html) rasters.

Results are plotted in from of graphs and maps in the directory [output](output).
Pre-computed example results are available in the directory [example_output](example_output).


## Dependencies

[Python 3.6](https://www.python.org/)
TODO


## Licence:
This code is licensed under the [MIT License](http://opensource.org/licenses/MIT) - see the LICENSE.md file for details

