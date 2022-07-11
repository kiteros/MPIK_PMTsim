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


#####
#
#doing the same parameter scan, but inverting it
#We scan on everz output space (K,P,R)
#For every selected output, 

noise_linspace = [0.5, 1, 2]#np.linspace(0.5, 2.0, num=1)
background_linspace = np.logspace(4, 6, num=3)
gain_linspace = np.linspace(4,16,num=5)


#For each element [N,B,G] we have a minimizing set
a_tensor = np.zeros((len(noise_linspace),len(background_linspace),len(gain_linspace),3))

prominence_linspace = np.linspace(3,6,num=5)
resampling_linspace = np.linspace(1,4,num=4)
kde_linspace = np.linspace(1,3, num=3)

global_minimum = 1000000
global_minimum_elements = [0,0,0,0]



for i2, prominence in enumerate(prominence_linspace):
	for window_ in ['tukey', 'blackman']:
		for j2, resampling_rate_ in enumerate(resampling_linspace):
			resampling_rate_ = int(resampling_rate_)
			for k2, kde_bandwidth in enumerate(kde_linspace):

				minimizing_metric = 0
				relative_ratios = []


				##We have one element of the output space

				for i, noise in enumerate(noise_linspace):


					for j, background_rate in enumerate(background_linspace):

						
						for k, gain in enumerate(gain_linspace):

							#one element of the input space
							

							ph = PeakHistogram(
								noise=noise,
								background_rate=background_rate,
								gain_linspace=[gain],
								graphs=False,
								prominence = prominence,
						        window = window_,
						        resampling_rate=resampling_rate_,
						        kde_banwidth=kde_bandwidth,
						        trace_lenght=2e5,

							)


							relative = ph.get_relative_gain_array()[0]
							minimizing_metric += abs(relative-1)
							relative_ratios.append(relative)


							
				minimizing_metric = np.sqrt(minimizing_metric)
				if minimizing_metric < global_minimum:
					global_minimum = minimizing_metric
					print("new minimizing metric: ", minimizing_metric)
					print("relative ratios average", np.mean(relative_ratios))
					global_minimum_elements[0] = prominence
					if window_ == 'tukey':
						global_minimum_elements[1] = 0
					elif window_ == 'blackman':
						global_minimum_elements[1] = 1
					global_minimum_elements[2] = resampling_rate_
					global_minimum_elements[3] = kde_bandwidth
					print("current best :",global_minimum_elements)
				





####Maybe we can do dichotomy ? or gradient descent

"""
with open('data/parameter_scan_all_space.npy', 'wb') as f:
	np.save(f, a_tensor)
"""

###lets try to represent it


plt.show()