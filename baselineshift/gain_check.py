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


def get_theoretical_gain(stddev, lamda, sigma, mu, baseline_mean, offset):
	
	tau = 2.821327162945735

	T_g = (4*stddev**2)/(tau*calculate_A(sigma,lamda,mu)*lamda*np.exp(sigma**2*lamda**2)*special.erfc(sigma*lamda)*(baseline_mean-offset))

	return T_g

esim = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e3,
    gain=10,
    no_signal_duration = 1e6,

    ps_mu = 15.11,
    ps_amp = 22.0,
    ps_lambda = 0.0659,
    ps_sigma = 2.7118,
)

pulse = Pulser(step=esim.t_step, pulse_type="none")
evts = pulse.generate_all()

gains = []
exp_gains = []


for g in np.linspace(2,25,num=15):

	####get the offset from a super low brate

	esim_offset = TraceSimulation(
	    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="../data/bb3_1700v_timing.txt",
	    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 1e3,
	    gain=10,
	    no_signal_duration = 1e4,

	    ps_mu = 15.11,
	    ps_amp = 22.0,
	    ps_lambda = 0.0659,
	    ps_sigma = 2.7118,
	)

	evts_br, k_evts = esim_offset.simulateBackground(evts)
	times, pmtSig, uncertainty_pmt = esim_offset.simulatePMTSignal(evts_br, k_evts)
	eleSig, uncertainty_ele = esim_offset.simulateElectronics(pmtSig, uncertainty_pmt, times)
	stimes, samples, samples_unpro, uncertainty_sampled = esim_offset.simulateADC(times, eleSig, uncertainty_ele, 1)
	offset, _, _, _, _, _, _, _, _, _ = esim_offset.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)

	esim_init = TraceSimulation(
	    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="../data/bb3_1700v_timing.txt",
	    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 3e10,
	    gain=g,
	    no_signal_duration = 1e4,

	    ps_mu = 15.11,
	    ps_amp = 22.0,
	    ps_lambda = 0.0659,
	    ps_sigma = 2.7118,
	)

	

	evts_br, k_evts = esim_init.simulateBackground(evts)
	times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts)
	eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)
	stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele, 1)
	bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
	

	exp_gain = get_theoretical_gain(stddev_mean, esim.ps_lambda, esim.ps_sigma, esim.ps_mu, bl_mean, offset)
	print(g)
	print(exp_gain)

	gains.append(g)
	exp_gains.append(exp_gain)


plt.figure()
plt.plot(gains, gains, label="True gain")
plt.plot(gains, exp_gains, label="Extracted gain")
plt.xlabel("Gain")
plt.ylabel("Gain")
plt.legend(loc="upper left")
plt.show()



