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

noise_linspace = [0.5, 1, 2]#np.linspace(0.5, 2.0, num=1)
background_linspace = np.logspace(4, 6, num=15)
gain_linspace = np.linspace(4,16,num=15)


#For each element [N,B,G] we have a minimizing set
a_tensor = np.zeros((len(noise_linspace),len(background_linspace),len(gain_linspace),3))

prominence_linspace = np.linspace(1,10,num=5)
resampling_linspace = np.linspace(1,8,num=5)
kde_linspace = np.linspace(1,3, num=4)

print(a_tensor)

#output_tensor = np.zeros((len(prominence_linspace), len(resampling_linspace), len(kde_linspace)))

for i, noise in enumerate(noise_linspace):


	for j, background_rate in enumerate(background_linspace):

		
		for k, gain in enumerate(gain_linspace):

			

			###for individual, just put linspace of one element
			
			absolute_dist_to_1 = 100
			minimizing_set = np.zeros(4)

			###lets attribute 0:tukey, and see for the others...

			###now being in this loop, the goal is to minimise this specific element (N,B,G) w.r to (P,W,R,K)
			#So we run over all those parameters, keep track of the ouput that is closest to 1, and output the best set of parameters.

			for i2, prominence in enumerate(prominence_linspace):
				for window_ in ['tukey']:
					for j2, resampling_rate_ in enumerate(resampling_linspace):
						resampling_rate_ = int(resampling_rate_)
						for k2, kde_bandwidth in enumerate(kde_linspace):


							ph = PeakHistogram(
									noise=noise,
									background_rate=background_rate,
									gain_linspace=[gain],
									graphs=False,
									prominence = prominence,
							        window = window_,
							        resampling_rate=resampling_rate_,
							        kde_banwidth=kde_bandwidth,
							        trace_lenght=1e5,

								)
							relative = ph.get_relative_gain_array()[0]
							if abs(relative-1) < absolute_dist_to_1:
								absolute_dist_to_1 = abs(relative-1)
								print("new minimizing element : ", relative)
								minimizing_set[0] = prominence
								minimizing_set[1] = 0
								minimizing_set[2] = resampling_rate_
								minimizing_set[3] = kde_bandwidth

			print("final : ")
			print(minimizing_set)

			a_tensor[i][j][k][0] = minimizing_set[0]
			a_tensor[i][j][k][1] = minimizing_set[2]
			a_tensor[i][j][k][2] = minimizing_set[3]


print(a_tensor)

####Maybe we can do dichotomy ? or gradient descent

with open('data/parameter_scan.npy', 'wb') as f:
	np.save(f, a_tensor)

###lets try to represent it


fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
plasma = cm.get_cmap('inferno', 12)

noise_array = []
br_array = []
gain_array = []
prominence_array = []
resampling_array = []
kde_array = []

for i, noise in enumerate(a_tensor):
	for j, br in enumerate(a_tensor[i]):
		for k, gain in enumerate(a_tensor[i][j]):
			color_offset = 0 #set to 0 after
			prominence_color = a_tensor[i][j][k][0]/(max(prominence_linspace)+color_offset)
			resampling_color = a_tensor[i][j][k][1]/(max(resampling_linspace)+color_offset)
			kde_color = a_tensor[i][j][k][2]/(max(kde_linspace)+color_offset)

			noise_array.append(noise_linspace[i])
			br_array.append(np.log10(background_linspace[j]))
			gain_array.append(gain_linspace[k])
			prominence_array.append(a_tensor[i][j][k][0])
			resampling_array.append(a_tensor[i][j][k][1])
			kde_array.append(a_tensor[i][j][k][2])



p = ax.scatter(noise_array, br_array, gain_array, c=prominence_array, marker='o', cmap=plasma)

ax.set_xlabel("Noise")
ax.set_ylabel("Background rate")
ax.set_zlabel("Gain")
ax.title.set_text("Prominence")

fig.colorbar(p)

ax = fig.add_subplot(2, 2, 2, projection='3d')
p = ax.scatter(noise_array, br_array, gain_array, c=resampling_array, marker='o', cmap=plasma)
ax.set_xlabel("Noise")
ax.set_ylabel("Background rate")
ax.set_zlabel("Gain")
ax.title.set_text("Resampling rate")

fig.colorbar(p)

ax = fig.add_subplot(2, 2, 3, projection='3d')
p = ax.scatter(noise_array, br_array, gain_array, c=kde_array, marker='o', cmap=plasma)
ax.set_xlabel("Noise")
ax.set_ylabel("Background rate")
ax.set_zlabel("Gain")
ax.title.set_text("Kde bandwidth")

fig.colorbar(p)

plt.show()