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

plt.rcParams['text.usetex'] = True

"""
Current best file for scan : By specifing of 3D parameter and output space, gives the best list of parameters and a heatmap of the space
"""
with open('../data/parameter_scan_3D_29jun_br5.5-6_g7-10.npy', 'rb') as file:

	prominence_values = np.load(file)
	resampling_values = np.load(file)
	kde_values = np.load(file)
	metric_array = np.load(file)

	print(prominence_values)


plt.figure()
plasma = cm.get_cmap('inferno', 12)
p = plt.scatter(prominence_values, kde_values, c=np.log10(metric_array), marker='s', cmap=plasma, s=100)

plt.xlabel(r"$\rho$",fontsize=20)
plt.ylabel(r"$B_W$",fontsize=20)

t = plt.colorbar(p)
t.set_label(r"$log(\gamma$)", rotation=90,fontsize=20)

plt.show()