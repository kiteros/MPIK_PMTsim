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

from scipy import signal
import scipy.special as sse


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

    f = 0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) 
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

def calculate_A(sigma, lamda, mu):
    """
    Fits the normalization coefficient A such that the maximum reached 1

    Parameters
    ----------
    sigma - pulse parameter
    lamda - pulse parameter
    mu - pulse position parameter

    Returns 
    ---
    A

    """
    A=1/(expnorm_normalized(find_mode(sigma,lamda,mu),lamda,sigma,mu))
    return A

def compute_I2(s, l, m, s_prime):
	##compute with the trapeze method this integral

	bounds = 1000

	A = calculate_A(s,l,m)

	mode = find_mode(s,l,m)

	m_prime = mode

	x = np.linspace(-bounds,bounds, num=10000)
	f = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-m_prime)**2)/(s_prime**2))
	h = f**2*g

	"""
	plt.figure()
	plt.plot(x, f)
	plt.show()
	"""

	inte = integrate.cumtrapz(h, x)[-1]

	return inte

def compute_I1(s, l, m, s_prime):
	##compute with the trapeze method this integral

	bounds = 1000

	A = calculate_A(s,l,m)
	mode = find_mode(s,l,m)

	m_prime = mode
	x = np.linspace(-bounds,bounds, num=10000)
	f = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-m_prime)**2)/(s_prime**2))
	h = f*g

	"""
	plt.figure()
	plt.plot(x, h)
	plt.show()
	"""

	inte = integrate.cumtrapz(h, x, initial=0)[-1]

	return inte


def expected_value(s, l, m, pulse):

	bounds = 1000
	A = calculate_A(s,l,m)
	x = np.linspace(-bounds,bounds, num=10000)

	n_trials = 1000
	results = []
	for i in range(n_trials):

		t = -norm.rvs(loc=0, scale=pulse.pulse_std, size=1)

		t = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*t))*sse.erfc((m+l*s*s-t)/(np.sqrt(2)*s))

		results.append(t)

	return np.mean(results)

####start by loading a pulser and making it act on the trace, print it

gain_linspace = np.linspace(4,30,num=50)


extracted_gain_mode = []
extracted_gain_ampdist = []

esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e6,
    gain=10,
    no_signal_duration = 1e4,
    noise=5,
)


pulser_init = Pulser(step=esim_init.t_step, duration=esim_init.no_signal_duration, pulse_type="pulsed")


I_1 = compute_I1(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init.pulse_std)
I_2 = compute_I2(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init.pulse_std)


for i in gain_linspace:

	esim = TraceSimulation(
	    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="../data/bb3_1700v_timing.txt",
	    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 1e6,
	    gain=i,
	    no_signal_duration = 5e5,
	    noise=1.5,
	)

	pulse = Pulser(step=esim.t_step, duration=esim.no_signal_duration, pulse_type="pulsed")
	evts = pulse.generate_all()

	
	evts_br, k_evts = esim.simulateBackground(evts)

	# pmt signal
	times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


	eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)


	# adc signal
	stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)



	###Now the goal is to identify every peak

	samples = signal.resample(samples, 4*len(stimes), window='boxcar')
	stimes = np.linspace(np.min(stimes), np.max(stimes),4*len(stimes), endpoint=False)

	maxs, _ = signal.find_peaks(samples, prominence=10) ###promeminence devrait dependre du gain ? Non
	max_values_stimes = stimes[maxs]
	max_values = samples[maxs]

	"""
	plt.figure()
	plt.plot(stimes, samples)
	plt.scatter(max_values_stimes, max_values, marker='o', color='red')
	plt.xlabel("Number of samples")
	plt.ylabel("LSB")
	"""


	###Do an histogram of the peaks
	hist, bins = np.histogram(max_values, density=True, bins=30)
	width = (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2

	"""
	plt.figure()
	plt.bar(center, hist, align='center', width=width)
	plt.title("gain : " + str(i))
	"""

	g_prime_ampdist = (np.std(max_values)**2/(np.mean(max_values)-esim.offset-10))*(esim.ampDist_drift/(esim_init.ampStddev**2+1)) * (I_1/I_2)

	g_prime_mode = (np.std(max_values)**2/(np.mean(max_values)-esim.offset-10))*(1/((esim_init.ampStddev**2+1)*esim.ampMode)) * (I_1/I_2)

	extracted_gain_mode.append(g_prime_mode)
	extracted_gain_ampdist.append(g_prime_ampdist)


plt.figure()
plt.plot(gain_linspace, gain_linspace, label="True gain")
plt.plot(gain_linspace, extracted_gain_mode, label="Extracted gain mode")
plt.plot(gain_linspace, extracted_gain_ampdist, label="Extracted gain ampdist")



plt.legend(loc="upper left")
plt.show()

