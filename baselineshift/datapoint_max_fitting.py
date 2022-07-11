#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate


import sys
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')



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


"""
Code used to test to find the peak of expmgaussians analytically
"""

def expnorm(x,l,s,m):

	f = 0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*special.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) 
	return f

def find_nearest(array,value):

	array = np.sort(array)  ###sort ascending
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return array[idx-1], len(array)-(idx-1)-1
	else:
		return array[idx], len(array)-idx-1

def erfcxinv(v):

	x = np.linspace(-10000,10000, num=1000000)
	y = special.erfcx(x)

	value, position = find_nearest(y, v)

	return x[position]

def find_mode(sigma, lamda, mu):
	pos = mu-np.sqrt(2)*sigma*erfcxinv((1/(lamda*sigma))*np.sqrt(2/np.pi))+sigma**2*lamda
	return pos

def calculate_A(sigma, lamda, mu):
	A=1/(expnorm(find_mode(sigma,lamda,mu),lamda,sigma,mu))
	return A


erfcxinv(0.1) ##########works !


###we can prob do it automatically

ymaxs = []
xmaxs = []
lamdas = np.linspace(0.01,3.0,num=50)



plt.figure()
for s in np.linspace(1.0, 1.0, num=1):

	xmaxs_l = []
	ymaxs_l = []
	for l in np.linspace(0.1,0.9,num=7):
		x = np.linspace(-10,20,num=1000)
		y = calculate_A(s,l,0)*expnorm(x,l,s,0)
		plt.plot(x,y)
		maxval = np.amax(y)
		xmax = np.argmax(y)
		#plt.scatter(x[xmax], maxval)
		plt.scatter(find_mode(s,l,0),calculate_A(s,l,0)*expnorm(find_mode(s,l,0),l,s,0))

		
		xmaxs_l.append(x[xmax])
		ymaxs_l.append(maxval)

	xmaxs.append(xmaxs_l)
	xmaxs_l = []
	ymaxs.append(ymaxs_l)
	ymaxs_l = []

plt.show()

plt.figure()
