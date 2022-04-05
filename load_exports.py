#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate
from pulser import Pulser
import scipy
import math 
from trace_simulation import TraceSimulation
from scipy.optimize import curve_fit
from scipy import odr
from pylab import *
import statistics
import os.path

from debug_fcts.bl_shift import BL_shift
from debug_fcts.bl_stddev import BL_stddev
from debug_fcts.under_c import Under_c
from debug_fcts.debug import Debug
#from debug_fcts.baseline import Baseline
from debug_fcts.pulse import Pulse

from scipy.stats import norm

from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import linear_model 

from calculate_gains import GainCalculator
import csv

for filename in os.listdir('exports/'):
    f = os.path.join('exports/', filename)
    # checking if it is a file
    if os.path.isfile(f):
        #Just add to an array

        if float(filename.split("B=")[1].split(";")[0]) > 2.76e6 and float(filename.split("B=")[1].split(";")[0]) < 2.92e6:

	        with open(f, 'rb') as file:

	        	stimes = np.load(file)
	        	samples = np.load(file)
	        	samples_unpro = np.zeros(samples.shape)
	        	uncertainty_sampled = np.load(file)
	        	bl_mean = np.load(file)
	        	std = np.load(file)
	        	stddev_mean = np.load(file)
	        	spike = np.load(file)
	        	s_mean = np.load(file)
	        	std_unpro = np.load(file)
	        	bl_mean_uncertainty = np.load(file)
	        	bl_array = np.load(file)
	        	stddev_uncert_mean = np.load(file)
	        	skew = np.load(file)
	        	adc_noise = np.load(file)
	        	true_gain = np.load(file)
	        	true_background_rate = np.load(file)

	        	plt.figure()
	        	plt.plot(stimes, samples, label="gain="+str(true_gain)+"br+"+str(true_background_rate)+", duration=1e6 ns")
	        	plt.plot(stimes, bl_array,label="baseline array, bl="+str(bl_mean)+"std:"+str(stddev_mean))
	        	plt.plot(stimes, np.repeat(stddev_mean+s_mean, len(stimes)), label="standard dev")
	        	plt.plot(stimes, np.repeat(bl_mean, len(stimes)), label="baseline mean")

	        	plt.xlabel("time ns")
	        	plt.ylabel("signal")

	        	plt.legend(loc="upper right")
plt.show()