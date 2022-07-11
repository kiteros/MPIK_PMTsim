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


###################
#Here we create a signal with different sample (length) and measure the decrease of standard variation uncertainty with sample size
#In a way that is the standard deviation of the standard deviation 
###############


esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e7,
    gain=3,
    no_signal_duration = 7e4,
    noise=0.8

)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

std_standard_deviation = []
std_current = []

trace_lenght = []
nb_samples = []

for length in np.logspace(1,4.5, num=10):


	for times in range(12):

		###generate n times the same signal with the same parameters, and measure the standard deviaiton of the sample

	    esim = TraceSimulation(
	        ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
	        timeSpec="../data/bb3_1700v_timing.txt",
	        pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
	        background_rate = 1e7,
	        gain=3,
	        no_signal_duration = 7e4,
	        noise=0.8,
	    )

	    evts_br, k_evts = esim.simulateBackground(evts)

	    # pmt signal
	    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


	    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

	    # adc signal
	    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

	    #This part should be done all the time, even when it is loaded
	    bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)


	    stimes = stimes[100:100+int(length)]
	    samples = samples[100:100+int(length)]




	    std_current.append(np.std(samples)**2)###variance of stimes

	    

	trace_lenght.append(length)


	nb_samples.append(length/esim.t_sample)
	std_standard_deviation.append(np.std(std_current))
	std_current = []


standard_dev = np.sqrt(esim_init.ampStddev**2*(esim_init.lamda+esim_init.lamda**2)+esim_init.lamda)*np.sqrt(esim_init.pulseShape_TotalSomation)*esim_init.gain+esim_init.noise

print(standard_dev)


plt.figure()
plt.semilogx(nb_samples, std_standard_deviation, label="True")
plt.semilogx(nb_samples, [2*standard_dev**2/np.sqrt(x) for x in nb_samples], label="theoretical")
plt.xlabel("Number of samples")
plt.ylabel("standard deviation of the Variance")
plt.legend(loc="upper left")
plt.show()

