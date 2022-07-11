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

"""
This file tests the list of parameters provided and looks at the standard deviation of the relative gain
"""


### input_ = [N,G,Br]
### output_ = [P,W,R,K]

z = [8.09025568, -2.75538668]

brlinspace = np.logspace(3,8,num=8)

gainlinspace = np.linspace(5,10,num=3)

noise_linspace = [0.8, 1.5, 2]


fig, ax = plt.subplots(1, 3,constrained_layout=True)

iter_ = 0

for noise in noise_linspace:
	
	for gain in gainlinspace:
		extracted_gains = []
		for background_rate_ in brlinspace:
			
			esim = TraceSimulation(
			    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="../data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = background_rate_,
			    gain=gain,
			    no_signal_duration = 1e4,
			    noise=noise,
			)

			pulse = Pulser(step=esim.t_step, duration=esim.no_signal_duration, pulse_type="pulsed")
			evts = pulse.generate_all()

			evts_br, k_evts = esim.simulateBackground(evts)

			# pmt signal
			times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


			eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

			stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

			maxs, _ = signal.find_peaks(samples, prominence=10) ###promeminence devrait dependre du gain ? Non
			max_values_stimes = stimes[maxs]
			max_values = samples[maxs]

			s = np.std(max_values)
			print(s)
			extracted_gain = (s-z[1])/(z[0])

			
			extracted_gains.append(extracted_gain/gain)

		ax[iter_].semilogx(brlinspace, extracted_gains, label="gain="+str(gain))
	ax[iter_].set_title("Noise = "+str(noise))

	ax[iter_].legend(loc="upper left")
	ax[iter_].set_xlabel("NSB rate")
	ax[iter_].set_ylabel("G'/G")

	iter_ += 1

plt.show()