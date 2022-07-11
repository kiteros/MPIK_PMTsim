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




def get_metric(prominenece, sampling, kde):
	noise_linspace = [1]#np.linspace(0.5, 2.0, num=1)
	background_linspace = np.logspace(4, 6, num=3)
	gain_linspace = np.linspace(4,16,num=4)
	minimizing_metric = 0
	#relative_ratios = []


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
					prominence = prominenece,
			        window = 'tukey',
			        resampling_rate=sampling,
			        kde_banwidth=kde,
			        trace_lenght=1e5,

				)


				relative = ph.get_relative_gain_array()[0]
				minimizing_metric += abs(relative-1)
				
	return minimizing_metric

#####
#
#doing the same parameter scan, but inverting it
#We scan on everz output space (K,P,R)
#For every selected output, 

noise_linspace = [1]#np.linspace(0.5, 2.0, num=1)
background_linspace = np.logspace(4, 6, num=3)
gain_linspace = np.linspace(4,16,num=4)

prominence_linspace = np.linspace(2,9,num=11)
resampling_linspace = np.linspace(1,4,num=4)
kde_linspace = np.linspace(0.7,3.5, num=11)

prominence_values = []
resampling_values = []
kde_values = []
metric_array = []


###Load the file from data
with open('../data_exports/parameter_scan_3D.npy', 'rb') as file:
	prominence_values = np.load(file)
	resampling_values = np.load(file)
	kde_values = np.load(file)
	metric_array = np.load(file)



X = []
Y = []
Z = []



for i in range(len(prominence_values)):
	if resampling_values[i] == 1:
		X.append(prominence_values[i])
		Y.append(kde_values[i])
		Z.append(metric_array[i])

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

X_linspace = np.unique(X)
Y_linspace = np.unique(Y)


plasma = cm.get_cmap('inferno', 12)




x, y = np.meshgrid(X_linspace, Y_linspace)
z = np.zeros(x.shape)


###We have to loop over every values of X and Y:
for i in range(len(x)):
	for j in range(len(x[i])):
		x_ = x[i][j]
		y_ = y[i][j]
		
		for k in range(len(X)):
			if X[k] == x_ and Y[k] == y_:
				z[i][j] = Z[k]



c = plt.pcolormesh(x, y, z, cmap='RdBu')
plt.colorbar(c)


###So we first want to find a minimum in the (Prominence, KDE) space

initial_step = [0.5,0.25]
random_iteration = 15
descent_iterations = 45

for i in range(random_iteration):
	####we take a random point on the 2D plane
	x_r = np.random.random_sample()*(np.max(X)-np.min(X))+np.min(X)
	y_r = np.random.random_sample()*(np.max(Y)-np.min(Y))+np.min(Y)

	plt.scatter(x_r, y_r, c='yellow')

	initial_metric = get_metric(x_r, 1,y_r)

	gamma_coeff = 1

	for i in range(descent_iterations):


		####get a little bit on the 4 sides
		x_r_up = x_r + gamma_coeff*initial_step[0]
		y_r_up = y_r

		x_r_down = x_r - gamma_coeff*initial_step[0]
		y_r_down = y_r

		y_r_right = y_r + gamma_coeff*initial_step[1]
		x_r_right = x_r 

		y_r_left = y_r - gamma_coeff*initial_step[1]
		x_r_left = x_r

		"""
		plt.scatter(x_r_up, y_r_up, c='black')
		plt.scatter(x_r_down, y_r_down, c='black')
		plt.scatter(x_r_right, y_r_right, c='black')
		plt.scatter(x_r_left, y_r_left, c='black')
		"""


		####check which one is the lowest
		###check the four cases separatly

		up_metric = get_metric(x_r_up, 1,y_r_up)
		down_metric = get_metric(x_r_down, 1,y_r_down)
		right_metric = get_metric(x_r_right, 1,y_r_right)
		left_metric = get_metric(x_r_left, 1,y_r_left)


		##all cases :

		if up_metric < down_metric and up_metric < right_metric and up_metric < left_metric:
			plt.scatter(x_r_up, y_r_up, c='green')
			plt.plot([x_r, x_r_up], [y_r, y_r_up],color='w')
			x_r = x_r_up
			y_r = y_r_up
			gamma_coeff = np.abs(initial_metric - up_metric)
			initial_metric = up_metric

		if down_metric < up_metric and down_metric < right_metric and down_metric < left_metric:
			plt.scatter(x_r_down, y_r_down, c='green')
			plt.plot([x_r, x_r_down], [y_r, y_r_down],color='w')
			x_r = x_r_down
			y_r = y_r_down
			gamma_coeff = np.abs(initial_metric - down_metric)
			initial_metric = down_metric

		if right_metric < down_metric and right_metric < up_metric and right_metric < left_metric:
			plt.scatter(x_r_right, y_r_right, c='green')
			plt.plot([x_r, x_r_right], [y_r, y_r_right],color='w')
			x_r = x_r_right
			y_r = y_r_right
			gamma_coeff = np.abs(initial_metric - right_metric)
			initial_metric = right_metric

		if left_metric < down_metric and left_metric < right_metric and left_metric < up_metric:
			plt.scatter(x_r_left, y_r_left, c='green')
			plt.plot([x_r, x_r_left], [y_r, y_r_left],color='w')
			x_r = x_r_left
			y_r = y_r_left
			gamma_coeff = np.abs(initial_metric - left_metric)
			initial_metric = left_metric


		###draw a line
		



plt.show()
