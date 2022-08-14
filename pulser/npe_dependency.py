#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
import os
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')

"""
print(__file__)
p1 = os.path.abspath(__file__+"/../../")
sys.path.insert(0, p1+"\\debug_fcts")
sys.path.insert(0, p1+"\\simulation")
sys.path.insert(0, p1+"\\darkcounts")
sys.path.insert(0, p1+"\\baselineshift")
"""


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

from scipy import signal
import scipy.special as sse

import matplotlib.patches as patches


plt.rcParams['text.usetex'] = True

def expnorm_normalized(x,l,s,m):
    """
    Fits an exponentially modified gaussian (EMG), normalized (A=1)

    Parameters
    ----------
    x - x-values array
    l - pulse parameter (tail)
    s - pulse parameter (pulse)
    m - position parameter

    Returns 
    ---
    EMG distribitoon y-array

    """

    f = (l/2)*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) 
    return f

def find_nearest(array,value):

    array = np.sort(array)  ###sort ascending
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], len(array)-(idx-1)-1
    else:
        return array[idx], len(array)-idx-1

def erfcxinv(v):

    """
    Inverse complementary error function scaled

    Parameters
    ----------
    v - input value

    Returns 
    ---
    output value

    """

    x = np.linspace(-10000,10000, num=1000000)
    y = sse.erfcx(x)

    value, position = find_nearest(y, v)

    return x[position]

def find_mode(sigma, lamda, mu):
    """
    Find the mode of an EMG

    Parameters
    ----------
    sigma - pulse parameter
    lamda - pulse tail parameter
    mu - pulse position parameter

    Returns 
    ---
    mode

    """
    pos = mu-np.sqrt(2)*sigma*erfcxinv((1/(lamda*sigma))*np.sqrt(2/np.pi))+sigma**2*lamda
    return pos

def compute_I2(s, l, m, s_prime):
	##compute with the trapeze method this integral

	bounds = 1000


	mode = find_mode(s, l, m)

	x = np.linspace(-bounds,bounds, num=10000)
	f = 0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*(mode-x)))*sse.erfc((m+l*s*s-(mode-x))/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x)**2)/(s_prime**2))
	h = f*g

	inte = integrate.cumtrapz(h, x)[-1]

	return inte

def compute_max_expected_value(s, l, m, s_prime):

	bounds_x = 1000
	bounds_t = 1000

	x_trials = np.linspace(-bounds_x, bounds_x, num=5000)

	minimizing_Eh = 0
	minimizing_Xh = 0

	for x in x_trials:

		t = np.linspace(-bounds_t,bounds_t, num=5000)
		f = 0.5*l*np.exp(0.5*l*(l*s*s-2*(x-t)))*sse.erfc((l*s*s-(x-t))/(np.sqrt(2)*s))  # exponential gaussian
		g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((t)**2)/(s_prime**2))
		h = f*g

		inte = integrate.cumtrapz(h, t)[-1]

		if inte > minimizing_Eh:
			minimizing_Eh = inte
			minimizing_Xh = x

	print("Best minimizing x : ", minimizing_Xh)
	print("best minimizing integral value", minimizing_Eh)
	return minimizing_Eh, minimizing_Xh

def get_distrib_at_infinity(s, l, m, s_prime, n_events):

    bounds_x = 1000
    bounds_t = 1000

    x_trials = np.linspace(-bounds_x, bounds_x, num=100000)
    distrib = []

    for x in x_trials:

        t = np.linspace(-bounds_t,bounds_t, num=100000)
        f = 0.5*l*np.exp(0.5*l*(l*s*s-2*(x-t)))*sse.erfc((l*s*s-(x-t))/(np.sqrt(2)*s))  # exponential gaussian
        g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((t)**2)/(s_prime**2))
        h = f*g

        inte = integrate.cumtrapz(h, t)[-1]
        distrib.append(inte*n_events)


    return distrib

    



events_linspace = np.linspace(1,50,num=50)

_lambda = 0.0659
_sigma = 2.7118


###
#time_dist of 0.75 ns

sigma_pulser_linspace = np.linspace(1, 4, num=3)

sigma_pulser = 3
mode = find_mode(_sigma, _lambda, 0)
max_expn = expnorm_normalized(mode, _lambda, _sigma, 0)

n_draws = 1

plt.figure()

for sigma_ in sigma_pulser_linspace:

    peaks = []

    for n_events in events_linspace:

        peaks_before_averaging = []

        for j in range(n_draws):

            n_events=int(n_events)

            
            ###lets first create our x space
            x = np.linspace(-200,200, num=10000)

            

            random_events = norm.rvs(loc=0, scale=sigma_, size=int(n_events))

            f = np.repeat(0.0, len(x))

            for i in range(n_events):

                time_random_event = random_events[i]+norm.rvs(loc=0, scale=0.75, size=1)

                exp_mg = expnorm_normalized(x, _lambda, _sigma, time_random_event)

                for j in range(len(f)):

                    f[j] += exp_mg[j]

            peaks_before_averaging.append(max(f)/(n_events*max_expn))


        peaks.append(np.mean(peaks_before_averaging))
        peaks_before_averaging = []



    exp_val, _ = compute_max_expected_value(_sigma, _lambda, 0, np.sqrt(sigma_**2+0.75**2))

    color = np.random.rand(3,)


    plt.plot(events_linspace, [x for x in peaks], label=r"$\sigma=$"+str(sigma_),color=color)
    plt.plot(events_linspace, np.repeat(exp_val/max_expn, len(events_linspace)), '--', color=color)

plt.legend(loc="lower left",fontsize=12)
plt.grid()
plt.xlabel(r"$\lambda_N$",fontsize=20)
plt.ylabel(r"$\frac{s(h)}{\lambda_N}$",fontsize=20)
plt.show()