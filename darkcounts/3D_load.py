#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys

sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')

from pulser import Pulser
import scipy
import math 
from trace_simulation import TraceSimulation
from scipy.optimize import curve_fit
from scipy import odr
from pylab import *
import statistics
import os.path

from bl_shift import BL_shift
from bl_stddev import BL_stddev
from under_c import Under_c
from debug import Debug
#from debug_fcts.baseline import Baseline
from pulse import Pulse

from scipy.stats import norm

from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import linear_model 

from calculate_gains import GainCalculator
import csv
import scipy.fftpack

from scipy import special

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from scipy import signal

from pulse_peak_histogram_2 import PeakHistogram
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Current best file for scan : By specifing of 3D parameter and output space, gives the best list of parameters and a heatmap of the space
"""
with open(f, 'rb') as file:

	stimes = np.load(file)
	samples = np.load(file)
	samples_unpro = np.zeros(samples.shape)
	uncertainty_sampled = np.load(file)
	bl_mean = np.load(file)
	std = np.load(file)
	stddev_mean = np.load(file)

plasma = cm.get_cmap('inferno', 12)
p = ax.scatter(prominence_values, resampling_values, kde_values, c=metric_array, marker='o', cmap=plasma)

ax.set_xlabel("prominence_values")
ax.set_ylabel("resampling_values rate")
ax.set_zlabel("kde_values")
ax.title.set_text("Prominence")

fig.colorbar(p)